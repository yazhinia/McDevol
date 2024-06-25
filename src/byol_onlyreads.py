#!/usr/bin/env python
""" run byol training """

import os
import time
import argparse
import copy
import logging
import numpy as np
import pandas as pd
from numpy import random as np_random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
import logging

# TODO: syncbatch normaliation

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * min(1.0, (self.last_epoch + 1) / self.total_iters) for base_lr in self.base_lrs]

# Combine both schedulers
class WarmUpThenScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_scheduler, main_scheduler):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_finished = False
        super().__init__(optimizer)
    
    def get_lr(self):
        if not self.warmup_finished:
            if self.warmup_scheduler.last_epoch < self.warmup_scheduler.total_iters:
                lr = self.warmup_scheduler.get_lr()
            else:
                self.warmup_finished = True
                lr = self.main_scheduler.get_lr()
                self.main_scheduler.last_epoch = self.warmup_scheduler.last_epoch - self.warmup_scheduler.total_iters
        else:
            lr = self.main_scheduler.get_lr()
        return lr
    
    def step(self, epoch=None):
        if not self.warmup_finished:
            self.warmup_scheduler.step(epoch)
        else:
            self.main_scheduler.step(epoch)

def normalize_counts(counts: np.ndarray):
    """ normalize count by mean division """
    counts_norm = counts / counts.mean(axis=1, keepdims=True)

    return counts_norm


def drawsample_frombinomial(counts, fraction_pi):
    """ augment data using binomial distribution """
    floor_counts = np_random.binomial(\
        np.floor(counts.detach().cpu().numpy()).astype(int), fraction_pi)
    ceil_counts = np_random.binomial(\
        np.ceil(counts.detach().cpu().numpy()).astype(int), fraction_pi)
    sample_counts = torch.from_numpy(\
        ((floor_counts + ceil_counts) / 2).astype(np.float32)).to(counts.device)

    return sample_counts

def split_dataset(dataset, flag_test):
    """ split dataset into training and validation. Also, test set (if flag_test is true)"""
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    if flag_test:
        val_size = np.ceil((total_size - train_size) / 2).astype(int)
        test_size = np.floor((total_size - train_size) / 2).astype(int)
        # return random_split(dataset, [train_size, val_size, test_size])
        return random_split(dataset, [train_size, total_size-train_size, 0])
    else:
        val_size = np.ceil(total_size - train_size).astype(int)
        return random_split(dataset, [train_size, val_size])


def MLP(dim, projection_size, hidden_size=4096):
    " return multiple linear perceptron layer "

    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.LeakyReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        """ update target network by moving average of online network """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class EarlyStopper:
    """ early stop the model when validation loss increases """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """ check if validation loss increases """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def target_update_moving_average(ema_updater, online_encode, \
    online_project, target_encode, target_project):
    for online_params1, online_params2, target_params1, target_params2 \
        in zip(online_encode.parameters(), online_project.parameters(), \
        target_encode.parameters(), target_project.parameters()):
        target_encode.data = ema_updater.update_average(target_params1.data, online_params1.data)
        target_project.data = ema_updater.update_average(target_params2.data, online_params2.data)

class BYOLmodel(nn.Module):
    """ train BYOL model """
    def __init__(
        self,
        args,
        seed: int = 0,
    ):
        if not args.reads.dtype == np.float32:
            args.reads = args.reads.astype(np.float32)

        torch.manual_seed(seed)

        super(BYOLmodel, self).__init__()

        # Initialize simple attributes
        self.usecuda = args.cuda
        self.ncontigs, self.nsamples = args.reads.shape
        print(self.nsamples, self.ncontigs, 'n samples and n contigs')
        self.nhidden = 1024
        self.nlatent = 512 # 256 # 
        self.dropout = 0.1
        self.num_workers= 4 if args.cuda else 1
        self.outdir = args.outdir
        self.logger = args.logger
        self.scheduler = None
        self.augmentsteps = [0.7, 0.6, 0.5, 0.3] # [0.9, 0.7, 0.6, 0.5, 0.3]
        self.nnindices = []
        projection_size = 256 # 512 # 128 #
        projection_hidden_size = 4096

        self.pairlinks = args.pairlinks
        self.marker = args.marker

        cindices = np.arange(self.ncontigs)
        indices_withpairs = np.unique(self.pairlinks.flatten())
        self.pairindices = torch.from_numpy(cindices[indices_withpairs])
        self.indim = self.nsamples + 1 + 512

        self.read_counts = torch.from_numpy(normalize_counts(args.reads))
        self.kmeraug1 = torch.from_numpy(args.kmeraug1)
        self.kmeraug2 = torch.from_numpy(args.kmeraug2)
        self.contigs_length = torch.from_numpy(args.length)
        self.rawread_counts = torch.from_numpy(args.reads)
        self.cindices = torch.from_numpy(cindices)
        self.dataset = TensorDataset(self.read_counts, self.rawread_counts,\
            self.contigs_length, self.cindices, self.kmeraug1, self.kmeraug2)
        self.dataset_train, self.dataset_val, \
            self.dataset_test = split_dataset(self.dataset, True)

        self.pairs_train, self.pairs_val, \
            self.pairs_test = split_dataset(self.pairlinks, True)

        self.online_encoder = nn.Sequential(
            nn.Linear(self.indim, self.nhidden),
            nn.BatchNorm1d(self.nhidden),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.nhidden, self.nhidden),
            nn.BatchNorm1d(self.nhidden),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.nhidden, self.nlatent) # latent layer
        )

        self.online_projector = MLP(self.nlatent, projection_size, projection_hidden_size)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.use_momentum = True
        self.target_encoder = None
        self.target_projector = None
        moving_average_decay = 0.99
        self.target_ema_updater = EMA(moving_average_decay)

        if self.usecuda:
            self.cuda()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # "cpu" # 
            self.device = device
            print('Using device:', device)

            #Additional Info when using cuda
            if self.device == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    def data_augment(self, rcounts, contigs_length, fraction_pi):
        """ augment read counts """
        rcounts = rcounts.clone().detach()
        if fraction_pi >= 0.5:
            condition = contigs_length>=3000
        else:
            condition = contigs_length>=7000
        rcounts[condition] = drawsample_frombinomial(rcounts[condition], fraction_pi)

        return normalize_counts(rcounts).to(contigs_length[0].device)

    def initialize_target_network(self):
        """ give target network """
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p, q in zip(self.target_encoder.parameters(), self.target_projector.parameters()):
            p.requires_grad = False
            q.requires_grad = False

    def update_moving_average(self):
        """ update target network by moving average """

        target_update_moving_average(self.target_ema_updater, self.online_encoder, \
            self.online_projector, self.target_encoder, self.target_projector)

    def forward(self, x, xt):
        """ forward BYOL """

        latent = self.online_encoder(x)

        z1_online = self.online_predictor(self.online_projector(latent))
        z2_online = self.online_predictor(self.online_projector(self.online_encoder(xt)))

        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(xt))
            z2_target =  self.target_projector(self.target_encoder(x))

        byol_loss = self.compute_loss(z1_online, z1_target.detach()) + \
            self.compute_loss(z2_online, z2_target.detach()) # to symmetrize the loss


        # # L1 regularization
        # l1_parameters = []
        # for param in self.online_encoder.parameters():
        #     l1_parameters.append(param.view(-1))

        # for param in self.online_projector.parameters():
        #     l1_parameters.append(param.view(-1))

        # for param in self.online_predictor.parameters():
        #     l1_parameters.append(param.view(-1))

        # l1_regloss = 1e-2 * torch.abs(torch.cat(l1_parameters)).sum()
        # byol_loss += l1_regloss

        # marker gene loss
        # cosine_sim_matrix = F.cosine_similarity(z1_online[self.marker[]])


        return latent, byol_loss

    def compute_loss(self, z1, z2):
        """ loss for BYOL """
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)

        return torch.mean(2 - 2 * (z1 * z2.detach()).sum(dim=-1))


    def process_batches(self,
        epoch: int,
        dataloader,
        training: bool,
        scheduler,
        *args):
        """ process batches """

        epoch_losses = []
        epoch_loss = 0.0

        for in_data in dataloader:

            if training:
                loss_array, latent_space, fraction_pi, optimizer = args
                optimizer.zero_grad()
                read_counts, rawread_counts, contigs_length, _, kmeraug1, kmeraug2 = in_data

            else:
                loss_array, latent_space = args

                read_counts, rawread_counts, contigs_length, _, kmeraug1, kmeraug2 = in_data

            if self.usecuda:
                read_counts = read_counts.cuda()
                contigs_length = contigs_length.cuda()
                rawread_counts = rawread_counts.cuda()
                kmeraug1 = kmeraug1.cuda()
                kmeraug2 = kmeraug2.cuda()

            if training:
                ### augmentation by fragmentation ###
                augmented_reads1 = self.data_augment(rawread_counts, \
                    contigs_length, fraction_pi)
                augmented_reads2 = self.data_augment(rawread_counts, \
                    contigs_length, fraction_pi)

                # if self.usecuda:
                #     augmented_reads1 = augmented_reads1.cuda()
                #     augmented_reads2 = augmented_reads2.cuda()
                #     kmeraug1 = kmeraug1.cuda()
                #     kmeraug2 = kmeraug2.cuda()

                rc_reads1 = torch.log(augmented_reads1.sum(axis=1))
                rc_reads2 = torch.log(augmented_reads2.sum(axis=1))
                latent, loss = \
                    self(torch.cat((augmented_reads1, rc_reads1[:,None], kmeraug1), 1), \
                        torch.cat((augmented_reads2, rc_reads2[:,None], kmeraug2), 1))

            else:
                rc_reads = torch.log(read_counts.sum(axis=1))
                latent, loss = \
                    self(torch.cat((read_counts, rc_reads[:,None], kmeraug1), 1), \
                        torch.cat((read_counts, rc_reads[:,None], kmeraug1), 1))

            loss_array.append(loss.data.item())
            latent_space.append(latent.cpu().detach().numpy())

            if training:
                loss.backward()
                optimizer.step()
                self.update_moving_average()

            epoch_loss += loss.detach().data.item()

        # scheduler.step()
        # print(epoch, scheduler.get_lr())

        epoch_losses.extend([epoch_loss])
        self.logger.info(f'{epoch}: byol loss={epoch_loss}')

    def process_batches_withpairs(self,
        epoch: int,
        dataloader,
        training: bool,
        scheduler,
        *args):
        """ process batches """

        epoch_losses = []
        epoch_loss = 0.0

        for in_data in dataloader:

            pairs = in_data
            pairindices_1 = pairs[:, 0]
            pairindices_2 = pairs[:, 1]

            if self.usecuda:
                pairindices_1, pairindices_2 = \
                pairindices_1.cuda(), pairindices_2.cuda()
            if training:
                loss_array, latent_space, _, optimizer = args
                optimizer.zero_grad()

            else:
                loss_array, latent_space = args


            
            if training:
                ### augmentation by fragmentation ###
                augmented_reads1 = self.read_counts[pairindices_1]
                augmented_reads2 = self.read_counts[pairindices_2]
                augmented_kmers1 = self.kmeraug1[pairindices_1]
                augmented_kmers2 = self.kmeraug2[pairindices_2]

                if self.usecuda:
                    augmented_reads1 = augmented_reads1.cuda()
                    augmented_reads2 = augmented_reads2.cuda()
                    augmented_kmers1 = augmented_kmers1.cuda()
                    augmented_kmers2 = augmented_kmers2.cuda()

                rc_reads1 = torch.log(augmented_reads1.sum(axis=1))
                rc_reads2 = torch.log(augmented_reads2.sum(axis=1))
                latent, loss = \
                    self(torch.cat((augmented_reads1, rc_reads1[:,None], augmented_kmers1), 1), \
                        torch.cat((augmented_reads2, rc_reads2[:,None], augmented_kmers2), 1))
            else:
                augmented_reads1 = self.read_counts[pairindices_1]
                augmented_kmers1 = self.kmeraug1[pairindices_1]

                if self.usecuda:
                    augmented_reads1 = augmented_reads1.cuda()
                    augmented_kmers1 = augmented_kmers1.cuda()

                rc_reads = torch.log(augmented_reads1.sum(axis=1))
                input1 = torch.cat((augmented_reads1, rc_reads[:,None], augmented_kmers1), 1)
                latent, loss = self(input1, input1)
            loss_array.append(loss.data.item())
            latent_space.append(latent.cpu().detach().numpy())

            if training:
                loss.backward()
                optimizer.step()
                self.update_moving_average()

            epoch_loss += loss.detach().data.item()

        # scheduler.step()
        # print(epoch, scheduler.get_lr())

        epoch_losses.extend([epoch_loss])
        self.logger.info(f'{epoch}: pair loss={epoch_loss}')



    def trainepoch(
        self,
        nepochs: int,
        dataloader_train,
        dataloader_val,
        dataloader_splittrain,
        dataloader_splitval,
        optimizer,
        batchsteps,
        scheduler
    ):
        """ training epoch """

        batch_size = 4096 #256

        loss_train = []
        loss_val = []

        fraction_pi = self.augmentsteps[0]

        counter = 1
        nepochs_1 = int(nepochs / 2)
        nepochs_2 = int(nepochs / 2)

        # augmentation by sampling
        with torch.autograd.detect_anomaly():
        # detect nan occurrence (only to for loop parts)

            # check_earlystop = EarlyStopper()
            # for epoch in range(nepochs):

            #     if epoch % 2 == 0:
            for epoch in range(nepochs_1):

                if epoch in batchsteps:
                    batch_size = batch_size * 2

                    if len(dataloader_train.dataset) > batch_size:
                        dataloader_train = DataLoader(dataset=dataloader_train.dataset, \
                        batch_size= batch_size, shuffle=True, drop_last=True, \
                        num_workers=self.num_workers, pin_memory=self.cuda)
                    if len(dataloader_val.dataset) > batch_size:
                        dataloader_val = DataLoader(dataset=dataloader_val.dataset, \
                        batch_size= batch_size, shuffle=True, drop_last=True, \
                        num_workers=self.num_workers, pin_memory=self.cuda)

                    fraction_pi = self.augmentsteps[counter]
                    counter += 1

                    # print(fraction_pi, 'fraction pi')

                # training
                self.train()
                latent_space_train = []

                # initialize target network
                self.initialize_target_network()

                self.process_batches(epoch, dataloader_train, \
                    True, scheduler, loss_train, latent_space_train, fraction_pi, optimizer)

                # testing
                self.eval()
                latent_space_val = []

                with torch.no_grad():
                    self.process_batches(epoch, dataloader_val, \
                    False, scheduler, loss_val, latent_space_val)

                # scheduler.step()
                # print(epoch, scheduler.get_lr(), batch_size)

                # print(loss_val, loss_val[-1], 'validation loss')
                # if check_earlystop.early_stop(loss_val[-1]):
                #     break

            # augmentation by split read mapping
            for epoch in range(nepochs_2):
            # if epoch % 2 != 0:
                # training
                self.train()
                latent_space_train = []

                # initialize target network
                self.initialize_target_network()

                self.process_batches_withpairs(epoch + nepochs_1, dataloader_splittrain, \
                    True, scheduler, loss_train, latent_space_train, fraction_pi, optimizer)

                # testing
                self.eval()
                latent_space_val = []

                with torch.no_grad():
                    self.process_batches_withpairs(epoch + nepochs_1, dataloader_splitval, \
                    False, scheduler, loss_val, latent_space_val)

                # scheduler.step()
                # print(epoch, scheduler.get_lr(), batch_size)

                # print(loss_val, loss_val[-1], 'validation loss')
                # if check_earlystop.early_stop(loss_val[-1]):
                #     break

        np.save(self.outdir + '/loss_train.npy', np.array(loss_train))
        np.save(self.outdir + '/loss_val.npy', np.array(loss_val))

        return None


    def trainmodel(
        self,
        nepochs: int = 400, #
        lrate: float = 3e-6,
        batchsteps: list = None,
        ):
        """ train medevol vae byol model """

        if batchsteps is None:
            batchsteps = [75, 100, 150] # [1, 2, 3, 4]# [500, 1000, 2000] #[50, 100, 150, 200] # [30, 50, 70, 100], #[10, 20, 30, 45],
        batchsteps_set = sorted(set(batchsteps))

        dataloader_train = DataLoader(dataset=self.dataset_train, \
            batch_size=4096, drop_last=True, shuffle=True, \
            num_workers=self.num_workers, pin_memory=self.cuda)
        dataloader_val = DataLoader(dataset=self.dataset_val, \
            batch_size=4096, drop_last=True, shuffle=True, \
            num_workers=self.num_workers, pin_memory=self.cuda)

        # split read mapping dataloader
        dataloader_splittrain = DataLoader(dataset=self.pairs_train, \
            batch_size= 4096 * 2, shuffle=True, drop_last=True, \
            num_workers=self.num_workers, pin_memory=self.cuda)
        dataloader_splitval = DataLoader(dataset=self.pairs_val, \
            batch_size= 4096 * 2, shuffle=True, drop_last=True, \
            num_workers=self.num_workers, pin_memory=self.cuda)

        optimizer = Adam(self.parameters(), lr=lrate, weight_decay=1e-1)

        warmup_epochs = 50
        warmup_scheduler = WarmUpLR(optimizer, total_iters=warmup_epochs * len(dataloader_train))

        # Define the main scheduler
        main_scheduler = CosineAnnealingLR(optimizer, T_max=100-warmup_epochs, eta_min=0)
        self.scheduler = WarmUpThenScheduler(optimizer, warmup_scheduler, main_scheduler)

        self.trainepoch(
            nepochs, dataloader_train, dataloader_val, dataloader_splittrain, dataloader_splitval,\
            optimizer, batchsteps_set, self.scheduler)

        torch.save(self.state_dict(), self.outdir + '/byol_model.pth')

    def testmodel(self):
        """ test model """
        self.eval()

        dataloader_test = DataLoader(dataset=self.dataset_test, batch_size=4096, \
            drop_last=True, shuffle=True, num_workers=self.num_workers, pin_memory=self.cuda)
        loss_test = []
        latent_space_test = []

        with torch.no_grad():
            self.process_batches(0, dataloader_test, \
            False, self.scheduler, loss_test, latent_space_test)

        # np.save(self.outdir + '/byol_test.npy', np.array(loss_test))

    def getlatent(self):
        """ get latent space after training """

        dataloader = DataLoader(dataset=self.dataset, batch_size=4096,
            shuffle=False, drop_last=False, \
            num_workers=self.num_workers, pin_memory=self.cuda)

        loss_test = []
        latent_space = []
        # initialize target network
        # self.initialize_target_network()
        self.eval()
        with torch.no_grad():
            self.process_batches(0, dataloader, \
            False, self.scheduler, loss_test, latent_space)

        return np.vstack(latent_space)

class LinearClassifier(nn.Module):
    """ Linear classifier """
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """ linear layer forward """
        return self.fc(x)


def train_linear_classifier(byol_model, whole_dataloader, train_loader, test_loader, num_classes, logger):
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu") # "cpu" #
    print(device, 'device inside linear classifier')
    # Assuming the feature dimension is 2048 for BYOL
    classifier = LinearClassifier(input_dim=byol_model.nlatent, \
        num_classes=num_classes+1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_lc = torch.optim.Adam(classifier.parameters(), lr=0.001)

    byol_model.to(device)
    byol_model.eval()

    # freeze model parameters
    for param in byol_model.parameters():
        param.requires_grad = False

    # Train classifier
    for epoch in range(300): # Number of epochs
        classifier.train()
        # byol_model.eval()
        for read_counts, kmer, labels in train_loader:
            read_counts, labels = read_counts.to(device), labels.to(device)
            kmer = kmer.to(device)
            rc_log = torch.log(read_counts.sum(axis=1))
            rc_log = rc_log.to(device)

            # Get representations from BYOL model
            with torch.no_grad():
                representations = byol_model.online_encoder(\
                    torch.cat((read_counts, rc_log[:,None], kmer),1)).detach()

            # Train classifier on these representations
            outputs = classifier(representations)
            # print(torch.max(outputs.data, 1), 'outputs')
            loss = criterion(outputs, labels)

            optimizer_lc.zero_grad()
            loss.backward()
            optimizer_lc.step()
        print(epoch+1, "=", loss.detach())
        logger.info(f"Epoch {epoch+1}, Loss: {loss.detach().item()}")

    # Evaluate classifier
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for read_counts, kmer, labels in test_loader:
            read_counts, labels = read_counts.to(device), labels.to(device)
            kmer = kmer.to(device)
            rc_log = torch.log(read_counts.sum(axis=1))
            rc_log = rc_log.to(device)
            representations = byol_model.online_encoder(\
                torch.cat((read_counts, rc_log[:,None], kmer),1)).detach()
            outputs = classifier(representations)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.info(f'Accuracy of the network on the \
        {len(test_loader)*4096} test data: {100 * correct / total} %')

    classifier.eval()
    correct = 0
    total = 0
    predicted_labels = []
    probabilities = []
    with torch.no_grad():
        for read_counts, kmer, labels in whole_dataloader:
            read_counts, labels = read_counts.to(device), labels.to(device)
            kmer = kmer.to(device)
            rc_log = torch.log(read_counts.sum(axis=1))
            rc_log = rc_log.to(device)
            representations = byol_model.online_encoder(\
                torch.cat((read_counts, rc_log[:,None], kmer),1)).detach()
            outputs = classifier(representations)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            prob = nn.functional.softmax(outputs.data, dim=1)
            prob, _ = torch.max(prob, 1)
            predicted_labels.extend(predicted.detach().cpu().numpy())
            probabilities.extend(prob.detach().cpu().numpy())
    # print(predicted_labels, 'predicted labels')
    np.save(byol_model.outdir + "/assignment.npy",np.vstack(predicted_labels).flatten())
    pd.DataFrame(np.vstack(probabilities).flatten()).to_csv(\
        byol_model.outdir + '/probabilities', header=None, sep='\t')
    logger.info(f'Accuracy of the network on the \
        {len(whole_dataloader)*4096} whole data: {100 * correct / total} %')


def main() -> None:

    """ BYOL for metagenome binning """
    start = time.time()
    parser = argparse.ArgumentParser(
        prog="mcdevol",
        description="BYOL for metagenome binning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s \
        --reads --length --names --otuids --marker --kmeraug1 --kmeragu2 --outdir [options]",
        add_help=False,
    )

    parser.add_argument("--reads", type=str, \
        help="read coverage matrix in npz format", required=True)
    parser.add_argument("--length", type=str, \
        help="length of contigs in bp", required=True)
    parser.add_argument("--names", type=str, \
        help="ids of contigs", required=True)
    parser.add_argument("--pairlinks", type=str, \
        help="provide pair links array", required=True)
    parser.add_argument("--otuids", type=str, \
        help="otuids of contigs", required=True)
    parser.add_argument("--marker", type=str, \
        help="marker genes hit", required=True)
    parser.add_argument("--kmer", type=str, \
        help='kmer embedding', required=True)
    parser.add_argument("--kmeraug1", type=str, \
        help='kmer embedding augment 1', required=True)
    parser.add_argument("--kmeraug2", type=str, \
        help='kmer embedding augment 2', required=True)
    parser.add_argument("--outdir", type=str, \
        help="output directory", required=True)
    parser.add_argument("--nlatent", type=int, \
        help="number of latent space")
    parser.add_argument("--cuda", \
        help="use GPU to train & cluster [False]", action="store_true")

    args = parser.parse_args()

    args.reads = np.load(args.reads, allow_pickle=True)['arr_0']
    args.length = np.load(args.length, allow_pickle=True)['arr_0']
    args.names = np.load(args.names, allow_pickle=True)['arr_0']
    args.pairlinks = np.load(args.pairlinks,allow_pickle='True')
    args.kmer = np.load(args.kmer, allow_pickle=True).astype(np.float32)
    args.kmeraug1 = np.load(args.kmeraug1, allow_pickle=True).astype(np.float32)
    args.kmeraug2 = np.load(args.kmeraug2, allow_pickle=True).astype(np.float32)

    args.outdir = os.path.join(args.outdir, '')
    print(args.outdir ,'output directory')

    try:
        if not os.path.exists(args.outdir):
            print('create output folder')
            os.makedirs(args.outdir)
    except RuntimeError as e:
        print(f'output directory already exist. Using it {e}')

    logging.basicConfig(format='%(asctime)s - %(message)s', \
    level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S',
    filename=args.outdir + '/byol_getassignment.log', filemode='w')
    args.logger = logging.getLogger()

    byol = BYOLmodel(args)
    total_params = sum(p.numel() for p in byol.parameters() if p.requires_grad)
    print(total_params, 'total parameters', flush=True)
    byol.trainmodel()
    # # # # byol.testmodel()
    latent = byol.getlatent()
    np.save(args.outdir + '/latent.npy', latent)
    latent = np.load(args.outdir + '/latent.npy')
    print(f"BYOL training is completed in {time.time() - start} seconds")
    # byol.load_state_dict(torch.load(args.outdir + '/byol_model.pth'), strict= False)

    args.otuids = pd.read_csv(args.otuids, header=None)
    unique_otu_ids = args.otuids[0].unique()
    otu_mapping = {otu_id: idx for idx, otu_id in enumerate(unique_otu_ids)}
    args.otuids[1] = args.otuids[0].map(otu_mapping)

    args.marker = pd.read_csv(args.marker, header=None, sep='\t')
    args.marker = dict(args.marker.groupby(1)[0].apply(list))

    labels = args.otuids[1].to_numpy()
    read_counts = torch.from_numpy(normalize_counts(args.reads))
    kmers = torch.from_numpy(args.kmer)
    print(kmers, 'argskmer')
    dataset = TensorDataset(read_counts, kmers, torch.from_numpy(labels))


    train_size = int(read_counts.shape[0] * 0.8)
    test_size = int(read_counts.shape[0] - train_size)
    dataset_train, dataset_test = torch.utils.data.random_split(dataset,[train_size,test_size])
    train_loader = DataLoader(dataset_train, batch_size=4096, shuffle=True,\
            num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_test,batch_size=4096, shuffle=False, \
            num_workers=4, pin_memory=True)

    whole_dataloader = DataLoader(dataset, batch_size=4096, shuffle=False,\
            num_workers=4, pin_memory=True)
    train_linear_classifier(byol, whole_dataloader, train_loader, test_loader, np.max(labels), args.logger)
    args.logger.info(f'{time.time()-start}, seconds to complete')
if __name__ == "__main__" :
    main()



# self.byol_online_encoder = nn.Sequential(
#             nn.Linear(self.indim, self.nhidden),
#             nn.BatchNorm1d(self.nhidden),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.nhidden, self.nhidden),
#             nn.BatchNorm1d(self.nhidden),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.nhidden, self.nlatent) # latent layer
#         )

#         self.byol_online_predictor = nn.Sequential(
#             nn.Linear(self.nlatent, self.nlatent),
#             nn.BatchNorm1d(self.nlatent),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.nlatent, self.nlatent),
#             nn.BatchNorm1d(self.nlatent),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.nlatent, self.nlatent),
#         )

#         self.byol_target_encoder = nn.Sequential(
#             nn.Linear(self.indim, self.nhidden),
#             nn.BatchNorm1d(self.nhidden),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.nhidden, self.nhidden),
#             nn.BatchNorm1d(self.nhidden),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.nhidden, self.nlatent) # latent layer
#         )

#         self.byol_online_decoder = nn.Sequential(
#             nn.Linear(self.nlatent, self.nhidden),
#             nn.BatchNorm1d(self.nhidden),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.nhidden, self.nhidden),
#             nn.BatchNorm1d(self.nhidden),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.nhidden, self.indim)
#         )

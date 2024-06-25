#!/usr/bin/env python
""" run byol training """

import os
import time
import argparse
import random
import copy
import logging
from datetime import datetime
from typing import Optional, IO
import numpy as np
from numpy import random as np_random
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging


def normalize_counts(counts: np.ndarray):
    """ normalize count by mean division """
    counts_norm = counts / counts.mean(axis=1, keepdims=True)

    return counts_norm

def normalize_abundance(counts: np.ndarray):
    epsilon = 1e-5
    counts_norm = (counts + epsilon) / np.max((counts + epsilon), axis=1, keepdims=True)

    return counts_norm

def normalize_kmers(kmers: np.ndarray):
    epsilon = 1
    kmers_norm = (kmers + epsilon) / np.sum((kmers + epsilon), axis=1, keepdims=True)

    return kmers_norm

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
    train_size = int(0.6 * total_size)
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
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
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
        in zip(online_encode.parameters(), online_project.parameters(), target_encode.parameters(), target_project.parameters()):
        target_encode.data = ema_updater.update_average(target_params1.data, online_params1.data)
        target_project.data = ema_updater.update_average(target_params2.data, online_params2.data)

class BYOLmodel(nn.Module):
    """ train BYOL model """
    def __init__(
        self,
        args,
        seed: int = 0,
    ):
        read_counts = args.reads
        kmer_counts = args.kmers
        kmers_left = args.kmers_left
        kmers_right = args.kmers_right
        contigs_length = args.length
        outdir = args.outdir
        cuda = args.cuda
        ncontigs, nsamples = read_counts.shape
        rc_reads = read_counts.mean(axis=1)
        rc_kmers = kmer_counts.mean(axis=1)
        dropout = 0.1
        nlatent = 32
        kmerweight = 0.1


        if not isinstance(read_counts, np.ndarray) or not isinstance(kmer_counts, np.ndarray):
            raise ValueError("read counts and kmer counts must be Numpy arrays")

        if len(read_counts) != len(kmer_counts):
            raise ValueError("\
            input number of contigs in read counts and kmer counts must be the same")

        if not read_counts.dtype == kmer_counts.dtype \
            == kmers_left.dtype == np.float32:
            read_counts = read_counts.astype(np.float32)
            kmer_counts = kmer_counts.astype(np.float32)
            rc_reads = rc_reads.astype(np.float32)
            rc_kmers = rc_kmers.astype(np.float32)
            kmers_left = kmers_left.astype(np.float32)
            kmers_right = kmers_right.astype(np.float32)

        if nlatent is None:
            nlatent = 32

        if nlatent < 1:
            raise ValueError(f"Minimum 1 latent neuron, not {nlatent}")

        if nsamples < 1:
            raise ValueError(f"nsamples must be > 0, not {nsamples}")

        if kmerweight is None:
            kmerweight = 0.1

        torch.manual_seed(seed)

        super(BYOLmodel, self).__init__()
        print(nlatent, 'latent dimension')
        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ncontigs = ncontigs
        self.nkmers = 256
        self.nhidden = 512
        self.nlatent = nlatent
        self.dropout = dropout
        self.num_workers= 4 if cuda else 1
        self.outdir = outdir
        self.logger = args.logger

        self.augmentsteps = [0.9, 0.8, 0.7, 0.6, 0.5]
        self.kmerweight = kmerweight
        self.nnindices = []
        print('kmer weight is set to', kmerweight)

        projection_size = 256
        projection_hidden_size = 4096

        self.indim = self.nsamples + self.nkmers + 1

        self.read_counts = torch.from_numpy(normalize_counts(read_counts))
        self.kmer_counts = torch.from_numpy(normalize_counts(kmer_counts))
        self.contigs_length = torch.from_numpy(contigs_length)
        self.rawread_counts = torch.from_numpy(read_counts)
        self.kmers_left = torch.from_numpy(normalize_counts(kmers_left))
        self.kmers_right = torch.from_numpy(normalize_counts(kmers_right))
        self.cindices = torch.arange(self.ncontigs)
        self.dataset = TensorDataset(self.read_counts, self.kmer_counts, \
                self.rawread_counts, self.kmers_left, \
                self.kmers_right, self.contigs_length, self.cindices)
        self.dataset_train, self.dataset_val, \
            self.dataset_test = split_dataset(self.dataset, True)

        self.pairlinks = args.pairlinks

        print(self.pairlinks, len(set(self.pairlinks.ravel().flatten())), 'pair links')
        self.mlphidden = 4096
        self.online_encoder = nn.Sequential(
            nn.Linear(self.indim, self.nhidden),
            nn.BatchNorm1d(self.nhidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.nhidden, self.nhidden),
            nn.BatchNorm1d(self.nhidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.nhidden, self.nlatent) # latent layer
        )

        self.online_projector = MLP(self.nlatent, projection_size, projection_hidden_size)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)
        self.use_momentum = True
        self.target_encoder = None
        self.target_projector = None
        moving_average_decay = 0.99
        self.target_ema_updater = EMA(moving_average_decay)

        self.device = 'cpu'
        if cuda:
            self.cuda()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device
            print('Using device:', device)

            #Additional Info when using cuda
            if self.device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    def data_augment(self, rcounts, contigs_length, fraction_pi):
        """ augment read counts """
        rcounts = rcounts.clone().detach()
        if fraction_pi >= 0.5:
            condition = contigs_length>=4000
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
        " update target network by moving average "
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


        # L1 regularization
        l1_parameters = []
        for param in self.online_encoder.parameters():
            l1_parameters.append(param.view(-1))

        for param in self.online_projector.parameters():
            l1_parameters.append(param.view(-1))

        for param in self.online_predictor.parameters():
            l1_parameters.append(param.view(-1))

        l1_regloss = 1e-2 * torch.abs(torch.cat(l1_parameters)).sum()
        byol_loss += l1_regloss

        return latent, byol_loss

    def compute_loss(self, z1, z2):
        """ loss for BYOL """
        z1 = F.normalize(z1, dim=-1, p=2)
        z2 = F.normalize(z2, dim=-1, p=2)

        return 2 - 2 * (z1 * z2.detach()).sum(dim=-1).mean()


    def process_batches(self,
        epoch: int,
        dataloader,
        training: bool,
        *args):
        """ process batches """

        epoch_losses = []
        epoch_loss = 0.0

        for in_data in dataloader:

            if training:
                loss_array, latent_space, fraction_pi, optimizer = args
                optimizer.zero_grad()
                read_counts, kmer_counts, rawread_counts, \
                kmers_left, kmers_right, contigs_length, cindices = in_data

                cindices_ = cindices.cpu().detach().numpy()
                pairlinks_first_col = self.pairlinks[:, 0]
                pairlinks_second_col = self.pairlinks[:, 1]
                matching_indices_first_col = np.isin(cindices_, pairlinks_first_col)
                matching_indices_second_col = np.isin(cindices_, pairlinks_second_col)
                matching_indices_combined = matching_indices_first_col | matching_indices_second_col
                nopairlink_indices = np.where(~matching_indices_combined)[0]
                pairlinks_firstmatch = np.isin(self.pairlinks[:,0], \
                    cindices_[matching_indices_first_col])
                pairlinks_secondmatch = np.isin(self.pairlinks[:,1], \
                    cindices_[matching_indices_second_col])
                pairlinks_indices1 = self.pairlinks[pairlinks_firstmatch]
                pairlinks_indices2 = self.pairlinks[pairlinks_secondmatch]
                pairlinks_indices2[:,[0,1]] = pairlinks_indices2[:,[1,0]]
                pairlinks = np.concatenate((pairlinks_indices1, pairlinks_indices2))
                rng = np.random.default_rng()
                pairlinks = rng.choice(pairlinks,size=in_data[0].shape[0]-len(nopairlink_indices), replace=False)

                # for i, c in enumerate(cindices_):
                #     first_inpair = np.nonzero(self.pairlinks[:,0]==c)[0]
                #     second_inpair = np.nonzero(self.pairlinks[:,1]==c)[0]
                #     if first_inpair.size > 0 and second_inpair.size > 0:
                #         selected_pair = random.choice(\
                #             [self.pairlinks[random.choice(first_inpair)],\
                #             self.pairlinks[random.choice(second_inpair)]])
                #     elif first_inpair.size >= 1:
                #         selected_pair = self.pairlinks[random.choice(first_inpair)]
                #     elif second_inpair.size >= 1:
                #         selected_pair = self.pairlinks[random.choice(second_inpair)]
                #     else:
                #         nopairlink_indices.append(i)
                #         continue
                #     pairlinks.append(selected_pair)

                # pairlinks = np.vstack(pairlinks)

            else:
                loss_array, latent_space = args

                read_counts, kmer_counts, rawread_counts, \
                    kmers_left, kmers_right, contigs_length, cindices = in_data

            if self.usecuda:
                read_counts = read_counts.cuda()
                kmer_counts = kmer_counts.cuda()
                contigs_length = contigs_length.cuda()
                rawread_counts = rawread_counts.cuda()
                kmers_left = kmers_left.cuda()
                kmers_right = kmers_right.cuda()

            if training:
                ### augmentation by fragmentation ###
                # non-paired contigs
                augmented_reads1 = self.data_augment(rawread_counts[nopairlink_indices], \
                    contigs_length[nopairlink_indices], fraction_pi)
                augmented_reads2 = self.data_augment(rawread_counts[nopairlink_indices], \
                    contigs_length[nopairlink_indices], fraction_pi)

                kmer_choice = [kmers_left[nopairlink_indices], kmers_right[nopairlink_indices]]
                random.shuffle(kmer_choice)

                augmented_kmers1 = kmer_choice[0]
                augmented_kmers2 = kmer_choice[1]

                # paired contigs
                if pairlinks.size > 0:
                    augmented_pair1 = self.read_counts[pairlinks[:,0]].to(augmented_reads1.device)
                    augmented_pair2 = self.read_counts[pairlinks[:,1]].to(augmented_reads1.device)
                    augmented_pairkmers1 = self.kmer_counts[pairlinks[:,0]].to(augmented_reads1.device)
                    augmented_pairkmers2 = self.kmer_counts[pairlinks[:,1]].to(augmented_reads1.device)
 
                    augmented_reads1 = torch.cat((augmented_pair1, augmented_reads1),0)
                    augmented_reads2 = torch.cat((augmented_pair2, augmented_reads2),0)
                    augmented_kmers1 = torch.cat((augmented_pairkmers1, augmented_kmers1),0)
                    augmented_kmers2 = torch.cat((augmented_pairkmers2, augmented_kmers2),0)

                if self.usecuda:
                    augmented_reads1 = augmented_reads1.cuda()
                    augmented_reads2 = augmented_reads2.cuda()
                    augmented_kmers1 = augmented_kmers1.cuda()
                    augmented_kmers2 = augmented_kmers2.cuda()

                rc_reads1 = torch.log(augmented_reads1.sum(axis=1))
                rc_reads2 = torch.log(augmented_reads2.sum(axis=1))
                latent, loss = \
                    self(torch.cat((augmented_reads1, augmented_kmers1, rc_reads1[:,None]), 1), \
                        torch.cat((augmented_reads2, augmented_kmers2, rc_reads2[:,None]), 1))

            else:
                rc_reads = torch.log(read_counts.sum(axis=1))
                latent, loss = \
                    self(torch.cat((read_counts, kmer_counts, rc_reads[:,None]), 1), \
                        torch.cat((read_counts, kmer_counts, rc_reads[:,None]), 1))

            loss_array.append(loss.data.item())
            latent_space.append(latent.cpu().detach().numpy())

            if training:
                loss.backward()
                optimizer.step()
                self.update_moving_average()

            epoch_loss += loss.data.item()

        epoch_losses.extend([epoch_loss])
        self.logger.info(f'{epoch}: byol loss={epoch_loss}')


    def trainepoch(
        self,
        nepochs: int,
        dataloader_train,
        dataloader_val,
        optimizer,
        batchsteps,
        scheduler
    ):

        """ training epoch """

        batch_size = 256

        loss_train = []
        loss_val = []

        fraction_pi = 0.5 # self.augmentsteps[0]

        counter = 1

        with torch.autograd.detect_anomaly():
        # detect nan occurrence (only to for loop parts)

            check_earlystop = EarlyStopper()
            for epoch in range(nepochs):

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

                    # fraction_pi = self.augmentsteps[counter]
                    # counter += 1

                    # print(fraction_pi, 'fraction pi')

                # training
                self.train()
                latent_space_train = []

                # initialize target network
                self.initialize_target_network()

                self.process_batches(epoch, dataloader_train, \
                    True, loss_train, latent_space_train, fraction_pi, optimizer)
                scheduler.step()
                print(epoch, scheduler.get_last_lr(), batch_size)

                # testing
                self.eval()
                latent_space_val = []

                with torch.no_grad():
                    self.process_batches(epoch, dataloader_val, \
                    False, loss_val, latent_space_val)

                # print(loss_val, loss_val[-1], 'validation loss')    
                # if check_earlystop.early_stop(loss_val[-1]):
                #     break
            
        np.save(self.outdir + '/loss_train.npy', np.array(loss_train))
        np.save(self.outdir + '/latent_space_train.npy', np.vstack(latent_space_train))
        np.save(self.outdir + '/loss_val.npy', np.array(loss_val))
        np.save(self.outdir + '/latent_space_val.npy', np.vstack(latent_space_val))

        return None


    def trainmodel(
        self,
        nepochs: int = 500,
        lrate: float = 3e-4,
        batchsteps: list = None,
        ):
        """ train medevol vae byol model """

        if batchsteps is None:
            batchsteps = [100,200,300] #[50, 100, 150, 200] # [30, 50, 70, 100], #[10, 20, 30, 45],
        batchsteps_set = sorted(set(batchsteps))

        # batchsteps_string = (
        #     ", ".join(map(str, batchsteps_set))
        #     if batchsteps_set
        #     else "None"
        # )
        # print("\tNetwork properties:", file=self.logfile)
        # print("\tCUDA:", self.usecuda, file=self.logfile)
        # print("\tDropout:", self.dropout, file=self.logfile)
        # print("\tN latent:", self.nlatent, file=self.logfile)
        # print("\n\tTraining properties:", file=self.logfile)
        # print("\tN epochs:", nepochs, file=self.logfile)
        # batchsteps_string = (
        #     ", ".join(map(str, batchsteps_set))
        #     if batchsteps_set
        #     else "None"
        # )
        # print("\tBatchsteps:", batchsteps_string, file=self.logfile)
        # print("\tLearning rate:", lrate, file=self.logfile)
        # print("\tN sequences:", self.ncontigs, file=self.logfile)
        # print("\tN samples:", self.nsamples, file=self.logfile, end="\n\n")

        dataloader_train = DataLoader(dataset=self.dataset_train, \
            batch_size=4096, drop_last=True, shuffle=True, \
            num_workers=self.num_workers, pin_memory=self.cuda)
        dataloader_val = DataLoader(dataset=self.dataset_val, \
            batch_size=4096, drop_last=True, shuffle=True, \
            num_workers=self.num_workers, pin_memory=self.cuda)

        optimizer = Adam(self.parameters(), lr=lrate, weight_decay=1e-1)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)

        self.trainepoch(
            nepochs, dataloader_train, dataloader_val, \
            optimizer, batchsteps_set, scheduler)

        torch.save(self.state_dict(), self.outdir + '/autoencoder_trainedmodel.pth')

    def testmodel(self):
        """ test model """
        self.eval()

        dataloader_test = DataLoader(dataset=self.dataset_test, batch_size=256, \
            drop_last=True, shuffle=True, num_workers=self.num_workers, pin_memory=self.cuda)
        loss_test = []
        latent_space_test = []

        with torch.no_grad():
            self.process_batches(0, dataloader_test, \
            False, loss_test, latent_space_test)

        np.save(self.outdir + '/loss_test.npy', np.array(loss_test))


    def getlatent(self):
        """ get latent space after training """

        dataloader = DataLoader(dataset=self.dataset, batch_size=256,
            shuffle=False, drop_last=False, \
            num_workers=self.num_workers, pin_memory=self.cuda)

        loss_test = []
        latent_space = []

        self.eval()
        with torch.no_grad():
            self.process_batches(0, dataloader, \
            False, loss_test, latent_space)

        return np.vstack(latent_space)
    

def main() -> None:

    """ BYOL for metagenome binning """
    start = time.time()
    parser = argparse.ArgumentParser(
        prog="mcdevol",
        description="BYOL for metagenome binning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s \
        --reads read_counts --kmers kmer_counts --kmers_augleft --kmers_augright --length --names --outdir [options]",
        add_help=False,
    )

    parser.add_argument("--reads", type=str, \
        help="read coverage matrix in npz format", required=True)
    parser.add_argument("--kmers", type=str, \
        help="kmer frequency matrix in npz format", required=True)
    parser.add_argument("--kmers_augleft", type=str, \
        help="provide augmented kmer counts (left)", required=True)
    parser.add_argument("--kmers_augright", type=str, \
        help="provide augmented kmer counts (right)", required=True)
    parser.add_argument("--pairlinks", type=str, \
        help="provide pair links array", required=True)
    parser.add_argument("--length", type=str, \
        help="length of contigs in bp", required=True)
    parser.add_argument("--names", type=str, \
        help="ids of contigs", required=True)
    parser.add_argument("--outdir", type=str, \
        help="output directory", required=True)
    parser.add_argument("--nlatent", type=int, \
        help="number of latent space")
    parser.add_argument("--kmerweight", type=float, \
        help="set kmerweight between 0.1 to 1")
    parser.add_argument("--cuda", \
        help="use GPU to train & cluster [False]", action="store_true")

    args = parser.parse_args()

    args.reads = np.load(args.reads, allow_pickle=True)['arr_0']
    # loading below is preprocessed kmer counts
    # (divide by 2, correction for repeat region)
    args.kmers = np.load(args.kmers, allow_pickle=True)['arr_0']
    args.length = np.load(args.length, allow_pickle=True)['arr_0']
    args.names = np.load(args.names, allow_pickle=True)['arr_0']
    # args.pair_links = np.load(args.pair_links, allow_pickle=True)
    args.kmers_left = np.load(args.kmers_augleft, allow_pickle=True)['arr_0']
    args.kmers_right = np.load(args.kmers_augright, allow_pickle=True)['arr_0']

    args.outdir = os.path.join(args.outdir, '')
    print(args.outdir ,'output directory')

    args.pairlinks = np.load(args.pairlinks,allow_pickle='True')

    try:
        if not os.path.exists(args.outdir):
            print('create output folder')
            os.makedirs(args.outdir)
    except RuntimeError as e:
        print(f'output directory already exist. Using it {e}')

    logging.basicConfig(format='%(asctime)s - %(message)s', \
    level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S',
    filename=args.outdir + '/byol.log', filemode='w')
    args.logger = logging.getLogger()

    byol = BYOLmodel(args)
    byol.trainmodel()
    # byol.testmodel()
    latent = byol.getlatent()
    np.save(args.outdir + '/latent_mu.npy', latent)

    print(f"metagenome binning is completed in {time.time() - start} seconds")


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

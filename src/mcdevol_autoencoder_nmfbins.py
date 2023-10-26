from __future__ import annotations
import torch
from typing import Optional, IO, Union
from pathlib import Path
from torch.utils.data.dataset import TensorDataset
from multiprocessing import Pool
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import softmax
from torch.optim import Adam
from torch import Tensor
from torch import nn
from math import log
import random
from scipy.stats import zscore
import pandas as pd 
import numpy as np
import time

from datetime import datetime
from functools import partial

def normalize_counts(counts, flag: int):
    
    if flag == 1:
        counts_norm = counts / (counts.mean(axis=1, keepdims=True) + 1e-09)
        return counts_norm

    else:
        counts_norm = counts / torch.mean(counts, 1 ,keepdim=True)
        return counts_norm

def drawsample_frombinomial(counts, fraction_pi):

    floor_counts = np.random.binomial(np.floor(counts.detach().cpu().numpy()).astype(int), fraction_pi.astype(np.float32))
    ceil_counts = np.random.binomial(np.ceil(counts.detach().cpu().numpy()).astype(int), fraction_pi.astype(np.float32))
    sample_counts = torch.from_numpy(((floor_counts + ceil_counts) / 2).astype(np.float32)).to(counts.device)

    return sample_counts

def drawsample_frombinomial_2d(counts, p_values):
    result_array = torch.empty((counts.shape), dtype=torch.float32)
    for i in range(counts.shape[0]):
        p = p_values[i]
        floor_counts = np.random.binomial(np.floor(counts[i].detach().cpu().numpy()).astype(int), p) ## taking floor and ceil following by averaging leads to slightly higher autmented counts than actual input counts and hence using only floor
        # ceil_counts = np.random.binomial(np.ceil(counts[i].detach().cpu().numpy()).astype(int), p)
        # row_values = torch.from_numpy(((floor_counts + ceil_counts) / 2).astype(np.float32)) 
        row_values = torch.from_numpy(floor_counts.astype(np.float32)) ### need to check with above scenario and get read_counts_aug2 as absolute difference
        result_array[i, :] = row_values
    return result_array.to(counts.device)

def write_log(logfile, epoch, len_data, training, epoch_losses):

    print(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), f'\tEpoch: {epoch}\t', end=" ", file=logfile,
    )
    for f in epoch_losses:
        print(
            f'{f/len_data:.6f}', end=" ",file=logfile, 
        )
    print(f'\ttraining:\t{training}',end='\n',file=logfile)


class mcdevol_AE(nn.Module):

    def __init__(
        self,
        args,
        nhiddens: Optional[list[int]] = None,
        seed: int = 0,
    ):

        read_counts = args.reads
        kmer_counts = args.kmers
        Zbc_matrix = args.Zbcmatrix
        outdir = args.outdir
        nlatent = args.nlatent
        cuda = args.cuda
        ncontigs, nsamples = read_counts.shape
        contigids_inbins = args.contigids_inbins
        kmers_all = args.kmers_all
        dropout = 0.1

        kmerweight = args.kmerweight


        if not isinstance(read_counts, np.ndarray) or not isinstance(kmer_counts, np.ndarray):
            raise ValueError("read counts and kmer counts must be Numpy arrays")

        if len(read_counts) != len(kmer_counts):
            raise ValueError("input number of contigs in read counts and kmer counts must be the same")

        if not (read_counts.dtype == kmer_counts.dtype == np.float32):
            read_counts = read_counts.astype(np.float32)
            kmer_counts = kmer_counts.astype(np.float32)

        if nlatent is None:
            nlatent = 32

        if nlatent < 1:
            raise ValueError(f"Minimum 1 latent neuron, not {nlatent}")

        if nsamples < 1:
            raise ValueError(f"nsamples must be > 0, not {nsamples}")

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.1 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError(f"Minimum 1 neuron per layer, not {min(nhiddens)}")

        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be 0 <= dropout < 1, not {dropout}")
        
        if kmerweight is None:
            kmerweight = 0.1

        # torch.manual_seed(seed)

        super(mcdevol_AE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ncontigs = ncontigs
        self.nkmers = 256
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout
        self.num_workers= 4 if cuda else 1
        self.outdir = outdir

        self.kmerweight = kmerweight

        self.indim = self.nsamples + self.nkmers + 1
        self.encoderL2indim = self.nhiddens[0]
      
        self.read_counts = torch.from_numpy(read_counts/read_counts.mean(axis=1, keepdims=True))
        self.kmer_counts = torch.from_numpy(kmer_counts/kmer_counts.mean(axis=1, keepdims=True))
        self.rawread_counts = torch.from_numpy(read_counts)
        self.contigids_inbins = contigids_inbins
        self.indices = torch.arange(self.read_counts.shape[0])
        self.kmers_all = torch.from_numpy(kmers_all.astype(np.float32))
        self.Zbc_matrix = Zbc_matrix

        self.dataset = TensorDataset(self.read_counts, self.kmer_counts, self.rawread_counts, self.indices)

        total_size = len(self.dataset)
        train_size = int(0.8 * total_size)
        val_size = np.ceil((total_size - train_size) / 2).astype(int)
        test_size = np.floor((total_size - train_size) / 2).astype(int)
        print(total_size, train_size, val_size, test_size, 'augmented input')

        self.dataset_train, self.dataset_val, self.dataset_test = random_split(self.dataset, [train_size, val_size, test_size])


        self.encoderL1 = nn.Sequential(
            nn.Linear(self.indim, self.nhiddens[0]),
            nn.BatchNorm1d(self.nhiddens[0]),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        self.encoderL2 = nn.Sequential(
            nn.Linear(self.encoderL2indim, self.nhiddens[0]),
            nn.BatchNorm1d(self.nhiddens[0]),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )

        # Latent layers
        self.latent = nn.Linear(self.nhiddens[0], self.nlatent)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.nlatent, self.nhiddens[0]),
            nn.BatchNorm1d(self.nhiddens[0]),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.nhiddens[0], self.nhiddens[0]),
            nn.BatchNorm1d(self.nhiddens[0]),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.nhiddens[0], self.indim)
        )

        if cuda:
            self.cuda()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Using device:', device)

            #Additional Info when using cuda
            if device.type == 'cuda':
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    def select_indices(self, ind, fraction_pi):
        selected_index = np.random.default_rng().choice(np.arange(len(ind)), np.ceil(len(ind)*fraction_pi).astype(int), replace=False)
        return selected_index
    

    def data_augmentation(self, read_counts, indices, fraction_pi):
        
        R_counts = read_counts.clone().detach()
        kmer_counts_aug = []
        read_counts_aug1 = []
        read_counts_aug2 = []

        ind_lists = [np.array(self.contigids_inbins[f]) for f in indices]
        selected_inds = [self.select_indices(i, fraction_pi) for i in ind_lists]
        binom_p = np.array([self.Zbc_matrix[f][ind].sum() / self.Zbc_matrix[f].sum() for f, ind in zip(indices, selected_inds)], dtype=np.float32)

        read_counts_aug1 = drawsample_frombinomial_2d(R_counts, binom_p)
        read_counts_aug2 = R_counts - read_counts_aug1 ### try with absolute difference if floor and ceil values are used in binomial sampling. Problem when we don't augment for single contig bins
        inds_noaugmentation = torch.where(read_counts_aug2.sum(axis=1)==0.0)[0]
        read_counts_aug2[inds_noaugmentation] = R_counts[inds_noaugmentation]
        kmer_counts_aug = [self.kmers_all[f[ind]].sum(axis=0) for f, ind in zip(ind_lists, selected_inds)]
        kmer_counts_aug = torch.vstack(kmer_counts_aug)

        # print(read_counts_aug1)
        # kmer_counts_aug = []
        # read_counts_aug1 = []
        # read_counts_aug2 = []
        # s = time.time()
        # for count, f in zip(R_counts, indices):
        #     ind_list = np.array(self.contigids_inbins[f])
        #     inds = np.random.default_rng().choice(np.arange(len(ind_list)), np.ceil(len(ind_list)*fraction_pi).astype(int), replace=False)
        #     binom_p = self.Zbc_matrix[f][inds].sum() / self.Zbc_matrix[f].sum()
        #     readcounts_1 = drawsample_frombinomial(count, binom_p)
        #     readcounts_2 = count - readcounts_1
        #     read_counts_aug1.append(readcounts_1)
        #     read_counts_aug2.append(readcounts_2)
        #     kmer_counts_aug.append(self.kmers_all[ind_list[inds]].sum(axis=0))

        # read_counts_aug1 = torch.vstack(read_counts_aug1)
        # read_counts_aug2 = torch.vstack(read_counts_aug2)
        # kmer_counts_aug = torch.vstack(kmer_counts_aug)
        # print(read_counts_aug1, 'second round')
        # print(time.time() - s, 'seconds in second round')
        
        return normalize_counts(read_counts_aug1, 0).to(read_counts.device), normalize_counts(read_counts_aug2, 0).to(read_counts.device), normalize_counts(kmer_counts_aug, 0).to(indices.device)
    
       

    def encode(self, reads_in, kmers_in):

        Rc_tensor = torch.log(reads_in.sum(axis=1))
        tensor = self.encoderL1(torch.cat((reads_in, kmers_in, Rc_tensor[:,None]), 1))
        tensor = self.encoderL2(tensor)
        latent = self.latent(tensor)
        
        return latent

    def decode(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        
        reconstruction = self.decoder(latent)
        reads_out = reconstruction.narrow(1, 0, self.nsamples)
        reads_out = softmax(reads_out, dim=1)
        kmers_out = reconstruction.narrow(1, self.nsamples, self.nkmers)
        kmers_out = softmax(kmers_out, dim=1)
            
        return [reads_out, kmers_out]

    def forward(self, *args):

        read_counts = args[0]
        kmer_counts = args[1]
        training = args[2]
        if training:
            fraction_pi = args[3]
            indices = args[4]
            augmented_reads1, augmented_reads2, augmented_kmers = self.data_augmentation(read_counts, indices, fraction_pi)

            if self.usecuda:
                augmented_reads1 = augmented_reads1.cuda()
                augmented_reads2 = augmented_reads2.cuda()
                augmented_kmers = augmented_kmers.cuda()
            latent = self.encode(augmented_reads1, augmented_kmers)
            return self.decode(latent=latent) + [latent, augmented_reads2]
        else:
            latent = self.encode(read_counts, kmer_counts)
            return self.decode(latent=latent) + [latent]

        
    
    def calc_readloss(self, *args):
        reads_in = args[0]
        reads_out = args[1]
        read_weights = 1
        read_loss = (reads_in * torch.log(reads_out + 1e-9)).sum(axis=1)
        loss = -read_weights * read_loss

        return loss.mean(), read_loss.mean()

    def calc_kmerloss(self, *args):
        kmers_in = args[0]
        kmers_out = args[1]
        kmer_weights = 1
        kmer_loss = (kmers_in * torch.log(kmers_out + 1e-9)).sum(axis=1)
        loss = -kmer_weights * kmer_loss

        return loss.mean(), kmer_loss.mean()



    def calc_loss(self, *args):
        
        loss_R, read_loss = self.calc_readloss(args[0], args[1])
        loss_K, kmer_loss = self.calc_kmerloss(args[2], args[3])
        loss = loss_R + self.kmerweight * loss_K

        return loss, read_loss, kmer_loss
        

    def process_batches(self, epoch, dataloader, logfile, training: bool, *args):

        epoch_losses = []
        epoch_loss = 0.0
        epoch_countloss = 0.0
        epoch_kmerloss = 0.0

        for read_counts, kmer_counts, rawread_counts, indices in dataloader:
        
            if training:
                loss_array, latent_space, fraction_pi, optimizer = args
                # read_counts = single_batch[0]
                # kmer_counts = single_batch[1]
                # Rc_reads = single_batch[3]
                # Rc_kmers = single_batch[4]
                optimizer.zero_grad()

            else:
                loss_array, latent_space = args

            if self.usecuda:
                read_counts = read_counts.cuda()
                kmer_counts = kmer_counts.cuda()
                rawread_counts = rawread_counts.cuda()

            if training:
                reads_out, kmers_out, latent, reads_aug = self(rawread_counts, kmer_counts, training, fraction_pi, indices)
                # if fraction_pi > 0.5:
                #     loss, countloss, kmerloss = self.calc_loss(
                #     reads_aug, reads_out, kmer_counts, kmers_out
                # )
                # else:
                #     loss, countloss, kmerloss = self.calc_loss(
                #     read_counts, reads_out, kmer_counts, kmers_out
                # )
                loss, countloss, kmerloss = self.calc_loss(
                    reads_aug, reads_out, kmer_counts, kmers_out
                )
            else:
                reads_out, kmers_out, latent = self(read_counts, kmer_counts, training)

                loss, countloss, kmerloss = self.calc_loss(
                    read_counts, reads_out, kmer_counts, kmers_out
                )

            loss_array.append(loss.data.item())
            latent_space.append(latent.cpu().detach().numpy())

            if training:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.data.item()
            epoch_countloss += countloss.data.item()
            epoch_kmerloss += kmerloss.data.item()

        epoch_losses.extend([epoch_loss, epoch_countloss, epoch_kmerloss])

        write_log(logfile, epoch, len(dataloader), training, epoch_losses)
        
        logfile.flush()



    def trainepoch(
        self,
        nepochs: int,
        optimizer,
        # scheduler,
        batchsteps: list[int],
        augmentsteps: list[float],
        logfile,
    ):

        batch_size = 256

        dataloader_train = DataLoader(dataset=self.dataset_train, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=self.num_workers, pin_memory=self.cuda)
        dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=self.num_workers, pin_memory=self.cuda)
        
        loss_train = []
        loss_val = []

        single_batch = next(iter(dataloader_train))

        fraction_pi = augmentsteps[0]

        counter = 0

        for epoch in range(nepochs):
           
            if epoch in batchsteps:
                batch_size = batch_size * 2

                if len(dataloader_train.dataset) > batch_size:
                    dataloader_train = DataLoader(dataset=dataloader_train.dataset, batch_size= batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers, pin_memory=self.cuda)
                if len(dataloader_val.dataset) > batch_size:
                    dataloader_val = DataLoader(dataset=dataloader_val.dataset, batch_size= batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers, pin_memory=self.cuda)
                
                fraction_pi = augmentsteps[counter]
                counter += 1

                print(fraction_pi, 'fraction pi')


            """ training """
            
            self.train()         
            latent_space_train = []
           
            self.process_batches(epoch, dataloader_train, logfile, True, loss_train, latent_space_train, fraction_pi, optimizer) #, single_batch)
            # scheduler.step() 

            # print(optimizer.param_groups[0]["lr"], 'learning rate')

            """ testing """

            self.eval()     
            latent_space_val = []
            
            with torch.no_grad():
                self.process_batches(epoch, dataloader_val, logfile, False, loss_val, latent_space_val)

        np.save(self.outdir + '/loss_train.npy', np.array(loss_train))
        np.save(self.outdir + '/latent_space_train.npy', np.vstack(latent_space_train))
 
        np.save(self.outdir + '/loss_val.npy', np.array(loss_val))
        np.save(self.outdir + '/latent_space_val.npy', np.vstack(latent_space_val))

        return None


    def trainmodel(
        self,
        nepochs: int = 300,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [25, 50, 75, 100], #[10, 20, 30, 45],
        augmentsteps: Optional[list[float]] = [0.9, 0.5, 0.5, 0.3],
        logfile: Optional[IO[str]] = None,
    ):
        
        if lrate < 0:
            raise ValueError(f"Learning rate must be positive, not {lrate}")

        if nepochs < 1:
            raise ValueError("Minimum 1 epoch, not {nepochs}")

        if batchsteps is None:
            batchsteps_set: set[int] = set()

        else:
            # First collect to list in order to allow all element types, then check that
            # they are integers
            batchsteps = list(batchsteps)
            if not all(isinstance(i, int) for i in batchsteps):
                raise ValueError("All elements of batchsteps must be integers")
            if max(batchsteps, default=0) >= nepochs:
                raise ValueError("Max batchsteps must not equal or exceed nepochs")
            batchsteps_set = sorted(set(batchsteps))
            print(batchsteps_set, 'batchsteps_set')
            

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tDropout:", self.dropout, file=logfile)
            print("\tN hidden:", ", ".join(map(str, self.nhiddens)), file=logfile)
            print("\tN latent:", self.nlatent, file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN epochs:", nepochs, file=logfile)
            batchsteps_string = (
                ", ".join(map(str, batchsteps_set))
                if batchsteps_set
                else "None"
            )
            print("\tBatchsteps:", batchsteps_string, file=logfile)
            print("\tLearning rate:", lrate, file=logfile)
            print("\tN sequences:", self.ncontigs, file=logfile)
            print("\tN samples:", self.nsamples, file=logfile, end="\n\n")

        optimizer = Adam(self.parameters(), lr=lrate)
        # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)

        self.trainepoch(
            nepochs, optimizer, batchsteps_set, augmentsteps, logfile
        )

        return None

    def testmodel(self, logfile: Optional[IO[str]] = None):

        self.eval()

        dataloader_test = DataLoader(dataset=self.dataset_test, batch_size=256, drop_last=True, shuffle=True, num_workers=self.num_workers, pin_memory=self.cuda)
        loss_test = []
        latent_space_test = []

        with torch.no_grad():
            self.process_batches(0, dataloader_test, logfile, False, loss_test, latent_space_test)

        np.save(self.outdir + '/loss_test.npy', np.array(loss_test))
        
        return None

    def getlatent(self, logfile: Optional[IO[str]] = None):

        torch.save(self.state_dict(), self.outdir + '/autoencoder_trainedmodel.pth')

        self.eval()

        dataloader = DataLoader(dataset=self.dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=self.cuda)

        loss_test = []
        latent_space = []

        with torch.no_grad():
            self.process_batches(0, dataloader, logfile, False, loss_test, latent_space)
        
        return np.vstack(latent_space)
    

    def latent_ofbins(self, binreads, binkmers, logfile: Optional[IO[str]] = None):
        
        binreads_tensor = torch.from_numpy(binreads.astype(np.float32))
        binkmers_tensor = torch.from_numpy(binkmers.astype(np.float32))
        self.eval()

        dataset = TensorDataset(binreads_tensor, binkmers_tensor, binreads_tensor, binkmers_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=self.cuda)

        loss_test = []
        latent_space = []

        with torch.no_grad():
            self.process_batches(0, dataloader, logfile, False, loss_test, latent_space)
        
        return np.vstack(latent_space)
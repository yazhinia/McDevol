from __future__ import annotations
from typing import Optional, IO, Union
from pathlib import Path
from torch.utils.data.dataset import TensorDataset
from torch.multiprocessing import Process
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
import torch

from datetime import datetime


def normalize_counts(counts: np.ndarray, znorm: int):
    print("normalize starts")
    if znorm == 1:
        counts_norm = zscore(counts)
        return counts_norm
    
    elif znorm == 2:
        counts_norm = counts / (counts.mean(axis=1, keepdims=True) + 1e-09)
        return counts_norm

    else:
        print(np.min(counts), 'inside normalize')
        counts_norm = counts / counts.mean(axis=1, keepdims=True)
        return counts_norm

def drawsample_frombinomial(counts, fraction_pi):
    floor_counts = np.random.binomial(np.floor(counts.detach().cpu().numpy()).astype(int), fraction_pi)
    ceil_counts = np.random.binomial(np.ceil(counts.detach().cpu().numpy()).astype(int), fraction_pi)

    sample_counts = torch.from_numpy(((floor_counts + ceil_counts) / 2).astype(np.float32)).to(counts.device)
    print(torch.min(sample_counts), torch.max(sample_counts))
    return sample_counts

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
        contigs_length = args.length
        codon_counts = args.codons
        outdir = args.outdir
        flag = args.flag
        nlatent = args.nlatent
        znorm = args.zscore
        cuda = args.cuda
        ncontigs, nsamples = read_counts.shape
        Rc_reads = read_counts.mean(axis=1)
        Rc_kmers = kmer_counts.mean(axis=1)
        dropout = 0.1

        kmerweight = args.kmerweight
        codonweight = args.codonweight

        kmer_counts_left = args.kmers_left
        kmer_counts_right = args.kmers_right

        if not isinstance(read_counts, np.ndarray) or not isinstance(kmer_counts, np.ndarray):
            raise ValueError("read counts and kmer counts must be Numpy arrays")

        if len(read_counts) != len(kmer_counts):
            raise ValueError("input number of contigs in read counts and kmer counts must be the same")

        if not (read_counts.dtype == kmer_counts.dtype == codon_counts.dtype == kmer_counts_left.dtype == np.float32):
            read_counts = read_counts.astype(np.float32)
            kmer_counts = kmer_counts.astype(np.float32)
            codon_counts = codon_counts.astype(np.float32)
            Rc_reads = Rc_reads.astype(np.float32)
            Rc_kmers = Rc_kmers.astype(np.float32)
            kmer_counts_left = kmer_counts_left.astype(np.float32)
            kmer_counts_right = kmer_counts_right.astype(np.float32)


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
        if codonweight is None:
            codonweight = 1.0

        # torch.manual_seed(seed)

        super(mcdevol_AE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ncontigs = ncontigs
        self.nkmers = 256
        self.ncodons = 62
        self.flag = flag
        self.znorm = znorm
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout
        self.num_workers= 4 if cuda else 1
        self.outdir = outdir

        self.kmerweight = kmerweight
        self.codonweight = codonweight

        print('kmer weight is set to', kmerweight)
        print('codon weight is set to', codonweight)

        # a and b factor coverage input
        self.a = 1
        self.b = 100 # 1 / np.mean(read_counts.sum(axis=1) / contigs_length)
        

        if flag == 'reads':
            self.indim = self.nsamples
            self.encoderL2indim = self.nhiddens[0] + 1

        elif flag == 'kmers':
            self.indim = self.nkmers + 1
            self.encoderL2indim = self.nhiddens[0]

        elif flag == 'threesources':
            self.indim = self.nsamples + self.nkmers + self.ncodons + 1
            self.encoderL2indim = self.nhiddens[0]

        else:
            self.indim = self.nsamples + self.nkmers + 1
            self.encoderL2indim = self.nhiddens[0]
      
        self.read_counts = torch.from_numpy(normalize_counts(read_counts, 0))
        self.kmer_counts = torch.from_numpy(normalize_counts(kmer_counts, 0))
        self.codon_counts = torch.from_numpy(normalize_counts(codon_counts, 2))
        self.contigs_length = torch.from_numpy(contigs_length)
        self.rawread_counts = torch.from_numpy(read_counts)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
        self.kmers_left = torch.from_numpy(normalize_counts(kmer_counts_left, 0))
        self.kmers_right = torch.from_numpy(normalize_counts(kmer_counts_right, 0))


        self.dataset = TensorDataset(self.read_counts, self.kmer_counts, self.codon_counts, self.contigs_length, self.rawread_counts, self.kmers_left, self.kmers_right)

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

    def data_augment(self, *args):
        
        R_counts = args[0].clone().detach()
        contigs_length = args[1]
        fraction_pi = args[2]

        if fraction_pi >= 0.5:
            condition = contigs_length>=4000
        else:
            condition = contigs_length>=7000

        R_counts[condition] = drawsample_frombinomial(R_counts[condition], fraction_pi)
        print(torch.min(R_counts), 'before normalize')
        print(torch.min(normalize_counts(R_counts, 0)), torch.max(normalize_counts(R_counts, 0)), 'R counts')
        return normalize_counts(R_counts, 0).to(args[0].device)

    def encode(self, *args):

        if self.flag == 'reads' or self.flag == 'kmers':
            input_tensor = args[0]
            augmented_length = args[1]
            Rc_tensor = torch.log(input_tensor.sum(axis=1))
            tensor = self.encoderL1(torch.cat((input_tensor, Rc_tensor[:,None]), 1))
            # # coverage = torch.log((reads_in.sum(axis=1) + self.a) / (augmented_length + self.b))
            # tensor = self.encoderL2((torch.cat((tensor, Rc_tensor[:,None], coverage[:,None]), 1)))
            tensor = self.encoderL2(tensor)

        elif self.flag == 'threesources':
            reads_in = args[0]
            kmers_in = args[1]
            codons_in = args[2]
            Rc_tensor = torch.log(reads_in.sum(axis=1))
            tensor = self.encoderL1(torch.cat((reads_in, kmers_in, codons_in, Rc_tensor[:,None]), 1))
            tensor = self.encoderL2(tensor)

        else:
            reads_in = args[0]
            kmers_in = args[1]
            # augmented_length = args[2]
            Rc_tensor = torch.log(reads_in.sum(axis=1))
            # coverage = torch.log((reads_in.sum(axis=1) + self.a) / (augmented_length + self.b))
            # coverage = torch.log(1 + (reads_in.sum(axis=1) / augmented_length))
            tensor = self.encoderL1(torch.cat((reads_in, kmers_in, Rc_tensor[:,None]), 1))
            tensor = self.encoderL2(tensor)
            # tensor = self.encoderL1(torch.cat((reads_in, kmers_in, Rc_tensor[:,None], coverage[:,None]), 1 ))
            # tensor = self.encoderL2(torch.cat((tensor, Rc_tensor[:,None], coverage[:,None]), 1))
        
        latent = self.latent(tensor)
        # print(torch.min(latent), torch.max(latent), torch.mean(latent), 'latent value range')
        return latent

    def decode(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        
        reconstruction = self.decoder(latent)

        if self.flag == 'reads':
            reads_out = softmax(reconstruction, dim=1)
            return [reads_out]
        
        elif self.flag == 'kmers':
            if self.znorm:
                kmers_out = reconstruction
            else:
                kmers_out = softmax(reconstruction, dim=1)
            
            return [kmers_out]
        
        elif self.flag == 'threesources':
            reads_out = reconstruction.narrow(1, 0, self.nsamples)
            reads_out = softmax(reads_out, dim=1)
            kmers_out = reconstruction.narrow(1, self.nsamples, self.nkmers)
            codons_out = reconstruction.narrow(1, self.nsamples+self.nkmers, self.ncodons)
            codons_out = softmax(codons_out, dim=1)

            if not self.znorm:
                kmers_out = softmax(kmers_out, dim=1)

            return [reads_out, kmers_out, codons_out]

        else:
            reads_out = reconstruction.narrow(1, 0, self.nsamples)
            kmers_out = reconstruction.narrow(1, self.nsamples, self.nkmers)
            reads_out = softmax(reads_out, dim=1)

            if not self.znorm:
                kmers_out = softmax(kmers_out, dim=1)
            
            return [reads_out, kmers_out]

    def forward(self, *args):

        if self.flag == 'reads':
            read_counts = args[0]
            contigs_length = args[1]
            training = args[2]
            fraction_pi = args[3]

            augmented_length = contigs_length * fraction_pi
            
            if training:
                augmented = self.data_augment(read_counts, contigs_length, fraction_pi)
                latent = self.encode(augmented, augmented_length)
            else:
                latent = self.encode(read_counts, contigs_length)

        elif self.flag == 'kmers':
            print('augmentation with kmers alone is not applied')
            pass
    
        elif self.flag == 'threesources':
            read_counts = args[0]
            kmer_counts = args[1]
            codon_counts = args[2]
            contigs_length = args[3]
            training = args[4]
            
            if training:
                fraction_pi = args[5]
                augmented_reads = self.data_augment(read_counts, contigs_length, fraction_pi)
                if self.usecuda:
                    augmented_reads = augmented_reads.cuda()
                latent = self.encode(augmented_reads, kmer_counts, codon_counts)
            else:
                latent = self.encode(read_counts, kmer_counts, codon_counts)

            
        else:
            read_counts = args[0]
            kmer_counts = args[1]
            contigs_length = args[2]
            training = args[3]
            if training:
                fraction_pi = args[4]
                augmented_length = contigs_length * fraction_pi
                augmented_reads = self.data_augment(read_counts, contigs_length, fraction_pi)
                print(torch.min(augmented_reads))
                if self.usecuda:
                    augmented_reads = augmented_reads.cuda()
                latent = self.encode(augmented_reads, kmer_counts, augmented_length)
            else:
                latent = self.encode(read_counts, kmer_counts, contigs_length)

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
        if self.znorm:
            kmer_weights = 1 / self.nkmers
            kmer_loss = (kmers_in - kmers_out).pow(2).sum(axis=1)
            loss = kmer_weights * kmer_loss
        else: 
            kmer_weights = 1
            kmer_loss = (kmers_in * torch.log(kmers_out + 1e-9)).sum(axis=1)
            loss = -kmer_weights * kmer_loss

        return loss.mean(), kmer_loss.mean()

    def calc_codonloss(self, *args):
        codons_in = args[0]
        codons_out = args[1]
        codon_weights = 1
        codon_loss = (codons_in * torch.log(codons_out + 1e-9)).sum(axis=1)
        loss = -codon_weights * codon_loss

        return loss.mean(), codon_loss.mean()


    def calc_loss(self, *args):
        
        if self.flag == 'reads':
            
            return self.calc_readloss(*args)
             
        elif self.flag == 'kmers':
            
            return self.calc_kmerloss(*args)
        
        elif self.flag == 'threesources':
            loss_R, read_loss = self.calc_readloss(args[0], args[1])
            loss_K, kmer_loss = self.calc_kmerloss(args[2], args[3])
            loss_C, codon_loss = self.calc_codonloss(args[4], args[5])
            loss = loss_R + (self.kmerweight * loss_K) + (self.codonweight *loss_C)

            return loss, read_loss, kmer_loss, codon_loss

        else:
            loss_R, read_loss = self.calc_readloss(args[0], args[1])
            loss_K, kmer_loss = self.calc_kmerloss(args[2], args[3])
            loss = loss_R + self.kmerweight * loss_K

            return loss, read_loss, kmer_loss
        

    def process_batches(self, epoch, dataloader, logfile, training: bool, *args):

        epoch_losses = []
        epoch_loss = 0.0
        epoch_countloss = 0.0


        if self.flag == 'threesources':
            epoch_kmerloss = 0.0
            epoch_codonloss = 0.0
        
        else:
            if self.flag != 'reads' and self.flag != 'kmers':
                epoch_kmerloss = 0.0

        for read_counts, kmer_counts, codon_counts, contigs_length, rawread_counts, kmers_left, kmers_right in dataloader:

            if training:
                loss_array, latent_space, fraction_pi, optimizer = args
                # read_counts = single_batch[0]
                # kmer_counts = single_batch[1]
                # codon_counts = single_batch[2]
                # Rc_reads = single_batch[3]
                # Rc_kmers = single_batch[4]
                optimizer.zero_grad()

                ## approach 1 ###
                if epoch <25 or epoch >50 and epoch <75:
                    kmers_aug_choice = kmers_left
                else:
                    kmers_aug_choice = kmers_right
                #################

                # ### approach 2 ###
                # if epoch <=25:
                #     kmers_aug_choice = kmer_counts
                # elif epoch >25 and epoch <= 175:
                #     kmers_aug_choice = kmers_left
                # else:
                #     kmers_aug_choice = kmers_right
                # ###################

                ### approach 3 ###
                # kmers_aug_choice = random.choice([kmers_left, kmers_right])
                ##################

                # ### approach 4 ###
                # if epoch <= 25:
                #     kmers_aug_choice = kmer_counts
                # else:
                #     kmers_aug_choice = random.choice([kmers_left, kmers_right])
                # ##################

                if self.usecuda:
                    kmers_aug_choice = kmers_aug_choice.cuda()

            else:
                loss_array, latent_space = args

            if self.usecuda:
                read_counts = read_counts.cuda()
                kmer_counts = kmer_counts.cuda()
                codon_counts = codon_counts.cuda()
                contigs_length = contigs_length.cuda()
                rawread_counts = rawread_counts.cuda()
                kmers_left = kmers_left.cuda()
                kmers_right = kmers_right.cuda()
                 
            if self.flag == 'reads':
                reads_out, latent = self(rawread_counts, contigs_length, training, fraction_pi)
                loss, countloss = self.calc_loss(
                    read_counts, reads_out
                )

            elif self.flag == 'kmers':               
                # kmers_out, latent = self(kmers_aug, contigs_length, training, fraction_pi)
                # loss, countloss = self.calc_loss(
                #     kmer_counts, kmers_out
                # )
                pass

            elif self.flag == 'threesources':
                if training:
                    reads_out, kmers_out, codons_out, latent = self(rawread_counts, kmers_aug_choice, codon_counts, contigs_length, training, fraction_pi)
                else:
                    reads_out, kmers_out, codons_out, latent = self(read_counts, kmer_counts, codon_counts, contigs_length, training)
                loss, countloss, kmerloss, codonloss = self.calc_loss(
                    read_counts, reads_out, kmer_counts, kmers_out, codon_counts, codons_out
                )

            else:
                if training:
                    reads_out, kmers_out, latent = self(rawread_counts, kmer_counts, contigs_length, training, fraction_pi)
                else:
                    reads_out, kmers_out, latent = self(read_counts, kmer_counts, contigs_length, training)

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
        
            if self.flag == 'threesources':
                epoch_kmerloss += kmerloss.data.item()
                epoch_codonloss += codonloss.data.item()
            else:
                if self.flag != 'reads' and self.flag != 'kmers':
                    epoch_kmerloss += kmerloss.data.item()

        epoch_losses.extend([epoch_loss, epoch_countloss])

        if self.flag == 'threesources':
            epoch_losses.extend([epoch_kmerloss, epoch_codonloss])
        
        else:
            if self.flag != 'reads' and self.flag != 'kmers':
                epoch_losses.extend([epoch_kmerloss])

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

        np.save(self.outdir + '/loss_train_'+str(self.codonweight)+'_.npy', np.array(loss_train))
        np.save(self.outdir + '/latent_space_train_'+str(self.codonweight)+'.npy', np.vstack(latent_space_train))
 
        np.save(self.outdir + '/loss_val_'+str(self.codonweight)+'_.npy', np.array(loss_val))
        np.save(self.outdir + '/latent_space_val_'+str(self.codonweight)+'_.npy', np.vstack(latent_space_val))

        return None


    def trainmodel(
        self,
        nepochs: int = 300,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [50, 100], #[10, 20, 30, 45],
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

        np.save(self.outdir + '/loss_test_'+str(self.codonweight)+'_.npy', np.array(loss_test))
        
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
    

    def latent_ofbins(self, binreads, binkmers, binlength, logfile: Optional[IO[str]] = None):
        
        binreads_tensor = torch.from_numpy(binreads.astype(np.float32))
        binkmers_tensor = torch.from_numpy(binkmers.astype(np.float32))
        binlength_tensor = torch.from_numpy(binlength)
        self.eval()

        dataset = TensorDataset(binreads_tensor, binkmers_tensor, binreads_tensor, binlength_tensor, binkmers_tensor, binlength_tensor,binlength_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=self.cuda)

        loss_test = []
        latent_space = []

        with torch.no_grad():
            self.process_batches(0, dataloader, logfile, False, loss_test, latent_space)
        
        return np.vstack(latent_space)
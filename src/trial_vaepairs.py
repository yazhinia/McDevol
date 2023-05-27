from __future__ import annotations
from typing import Optional, IO, Union
from pathlib import Path
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from torch.optim import Adam
from torch import Tensor
from torch import nn
from math import log
import pandas as pd 
import numpy as np
import torch

# import vamb.vambtools as _vambtools

def get_randompairs(cluster_ids):
    
    random_clusterpairs = []
    cluster_counts = np.concatenate((np.arange(np.max(cluster_ids)+1).reshape(-1,1),np.bincount(cluster_ids).reshape(-1,1)), axis=1)

    for f in range(cluster_counts.shape[0]):
        cluster_id = cluster_counts[f][0]
        contig_list = np.nonzero(cluster_ids==cluster_id)[0]
        cluster_size = len(contig_list)

        if cluster_size > 1:
            tensor_contig_list = torch.tensor(contig_list)
            pairs = torch.cartesian_prod(tensor_contig_list, tensor_contig_list)
            random = torch.rand(pairs.shape[0])
            pairs = pairs[random < cluster_size / (cluster_size**2 - cluster_size)]
            random_clusterpairs.append(pairs)
        else:
            random_clusterpairs.append(torch.tensor([contig_list[0], contig_list[0]]))

    return torch.vstack(random_clusterpairs)


def normalize_counts(counts: np.ndarray):

    # counts /= counts.sum(axis=0, keepdims=True)
    counts /= counts.sum(axis=1, keepdims=True)

    return counts.astype(np.float32)


def make_dataloader(
    rpkm: np.ndarray,
    tnf: np.ndarray,
    lengths: np.ndarray,
    batchsize: int = 256,
    destroy: bool = False,
    cuda: bool = False,
) -> tuple[DataLoader[tuple[Tensor, Tensor, Tensor]], np.ndarray]:
    """Create a DataLoader and a contig mask from RPKM and TNF.

    The dataloader is an object feeding minibatches of contigs to the VAE.
    The data are normalized versions of the input datasets, with zero-contigs,
    i.e. contigs where a row in either TNF or RPKM are all zeros, removed.
    The mask is a boolean mask designating which contigs have been kept.

    Inputs:
        rpkm: RPKM matrix (N_contigs x N_samples)
        tnf: TNF matrix (N_contigs x N_TNF)
        batchsize: Starting size of minibatches for dataloader
        destroy: Mutate rpkm and tnf array in-place instead of making a copy.
        cuda: Pagelock memory of dataloader (use when using GPU acceleration)

    Outputs:
        DataLoader: An object feeding data to the VAE
        mask: A boolean mask of which contigs are kept
    """

    if not isinstance(rpkm, np.ndarray) or not isinstance(tnf, np.ndarray):
        raise ValueError("TNF and RPKM must be Numpy arrays")

    if batchsize < 1:
        raise ValueError(f"Batch size must be minimum 1, not {batchsize}")

    if len(rpkm) != len(tnf) or len(tnf) != len(lengths):
        raise ValueError("Lengths of RPKM, TNF and lengths arrays must be the same")

    if not (rpkm.dtype == tnf.dtype == np.float32):
        raise ValueError("TNF and RPKM must be Numpy arrays of dtype float32")

    ### Copy arrays and mask them ###
    # Copy if not destroy - this way we can have all following operations in-place
    # for simplicity
    if not destroy:
        rpkm = rpkm.copy()
        tnf = tnf.copy()

    # Normalize samples to have same depth
    sample_depths_sum = rpkm.sum(axis=0)
    if np.any(sample_depths_sum == 0):
        raise ValueError(
            "One or more samples have zero depth in all sequences, so cannot be depth normalized"
        )
    rpkm *= 1_000_000 / sample_depths_sum

    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    mask = tnf.sum(axis=1) != 0
    depthssum = None
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        # mask &= depthssum != 0
        # depthssum = depthssum[mask]
        assert isinstance(depthssum, np.ndarray)

    # if mask.sum() < batchsize:
    #     raise ValueError(
    #         "Fewer sequences left after filtering than the batch size. "
    #         + "This probably means you try to run on a too small dataset (below ~10k sequences), "
    #         + "or that nearly all sequences were filtered away. Check the log file, "
    #         + "and verify BAM file content is sensible."
    #     )

    # _vambtools.numpy_inplace_maskarray(rpkm, mask)
    # _vambtools.numpy_inplace_maskarray(tnf, mask)

    # If multiple samples, normalize to sum to 1, else zscore normalize
    # if rpkm.shape[1] > 1:
    #     assert depthssum is not None  # we set it so just above
    #     rpkm /= depthssum.reshape((-1, 1))
    # else:
    #     _vambtools.zscore(rpkm, axis=0, inplace=True)

    # # Normalize TNF
    # _vambtools.zscore(tnf, axis=0, inplace=True)

    # Create weights
    lengths = lengths.astype(np.float32)
    # lengths = (lengths[mask]).astype(_np.float32)
    # weights = _np.log(lengths).astype(_np.float32) - 5.0
    # weights[weights < 2.0] = 2.0
    # weights *= len(weights) / weights.sum()
    # weights.shape = (len(weights), 1)

    ### Create final tensors and dataloader ###
    depthstensor = torch.from_numpy(rpkm)  # this is a no-copy operation
    tnftensor = torch.from_numpy(tnf)
    lengthstensor = torch.from_numpy(lengths)

    n_workers = 4 if cuda else 1
    dataset = TensorDataset(depthstensor, tnftensor, lengthstensor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=True,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
    )

    return dataloader, mask

def set_batchsize(
    data_loader: DataLoader, batch_size: int, encode=False
) -> DataLoader:
    """Effectively copy the data loader, but with a different batch size.

    The `encode` option is used to copy the dataloader to use before encoding.
    This will not drop the last minibatch whose size may not be the requested
    batch size, and will also not shuffle the data.
    """
    return DataLoader(
        dataset=data_loader.dataset,
        batch_size=batch_size,
        shuffle=not encode,
        drop_last=not encode,
        num_workers=1 if encode else data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
    )


# class vae_pair(nn.Module):
class VAE(nn.Module):

    def __init__(
        self,
        read_counts: np.ndarray,
        kmer_counts: np.ndarray,
        lengths: np.ndarray,
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
        seed: int = 0,
    ):
        ncontigs, nsamples = read_counts.shape
        Rc_reads = read_counts.sum(axis=1)

        if not isinstance(read_counts, np.ndarray) or not isinstance(kmer_counts, np.ndarray):
            raise ValueError("read counts and kmer counts must be Numpy arrays")

        if len(read_counts) != len(kmer_counts) or len(kmer_counts) != len(lengths):
            raise ValueError("Lengths of read counts, kmer counts and lengths arrays must be the same")

        if not (read_counts.dtype == kmer_counts.dtype == np.float32):
            read_counts = read_counts.astype(np.float32)
            kmer_counts = kmer_counts.astype(np.float32)
            Rc_reads = Rc_reads.astype(np.float32)

        if nlatent < 1:
            raise ValueError(f"Minimum 1 latent neuron, not {nlatent}")

        if nsamples < 1:
            raise ValueError(f"nsamples must be > 0, not {nsamples}")

        if alpha is None:
            alpha = 0.15 if nsamples > 1 else 0.50

        if nhiddens is None:
            nhiddens = [512, 512] if nsamples > 1 else [256, 256]

        if dropout is None:
            dropout = 0.2 if nsamples > 1 else 0.0

        if any(i < 1 for i in nhiddens):
            raise ValueError(f"Minimum 1 neuron per layer, not {min(nhiddens)}")

        if beta <= 0:
            raise ValueError(f"beta must be > 0, not {beta}")

        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be 0 < alpha < 1, not {alpha}")

        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be 0 <= dropout < 1, not {dropout}")

        torch.manual_seed(seed)

        super(VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ncontigs = ncontigs
        self.nkmers = 256 # 103
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout
        self.num_workers= 4 if cuda else 1

        # read_counts = read_counts * 300 / lengths[:,None]
        self.read_counts = torch.from_numpy(normalize_counts(read_counts))
        self.kmer_counts = torch.from_numpy(normalize_counts(kmer_counts))
        self.Rc_reads = torch.from_numpy(Rc_reads)
        self.lengths = torch.from_numpy(lengths - 3) # account for tetramer length, lengths=l-k+1

        # encoding with two fully connected linear layers
        self.encoder_fc1 = nn.Linear(self.nsamples + self.nkmers + 2, self.nhiddens[0])
        self.encoder_fc2 = nn.Linear(self.nhiddens[0] + 2, self.nhiddens[0])      
        self.encodernorm = nn.BatchNorm1d(self.nhiddens[0]) 

        # Latent layers
        self.mu = nn.Linear(self.nhiddens[0], self.nlatent)
        self.logvar = nn.Linear(self.nhiddens[0], self.nlatent)

        # decoding with two fully connected linear layers
        self.decoder_fc1 = nn.Linear(self.nlatent + 2, self.nhiddens[0])
        self.decoder_fc2 = nn.Linear(self.nhiddens[0] + 2, self.nhiddens[0])
        self.decodernorm = nn.BatchNorm1d(self.nhiddens[0]) 
            
        # Reconstruction (output) layer
        self.outputlayer = nn.Linear(self.nhiddens[0], self.nsamples + self.nkmers)

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.dropoutlayer = nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()


    def _encode(self, input_tensor: Tensor, Rc_tensor: Tensor, lengths_tensor: Tensor) -> tuple[Tensor, Tensor]:

        tensor = self.encodernorm(self.dropoutlayer(self.relu(self.encoder_fc1(torch.cat((input_tensor, Rc_tensor[:,None], lengths_tensor[:,None]), 1)))))
        tensor = self.encodernorm(self.dropoutlayer(self.relu(self.encoder_fc2(torch.cat((tensor, Rc_tensor[:,None], lengths_tensor[:,None]), 1)))))
        
        mu = self.mu(tensor)
        
        logvar = self.logvar(tensor)
        
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        
        epsilon = torch.randn(mu.size(0), mu.size(1))

        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True

        latent = mu + epsilon * torch.exp(logvar/2)

        return latent

    def _decode(self, latent_tensor: Tensor, Rc_xp_tensor: Tensor, lengths_xp_tensor: Tensor) -> tuple[Tensor, Tensor]:
        
        tensor = self.encodernorm(self.dropoutlayer(self.relu(self.decoder_fc1(torch.cat((latent_tensor, Rc_xp_tensor[:,None], lengths_xp_tensor[:,None]), 1)))))
        tensor = self.encodernorm(self.dropoutlayer(self.relu(self.decoder_fc2(torch.cat((tensor, Rc_xp_tensor[:,None], lengths_xp_tensor[:,None]), 1)))))
        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and kmer signal
        reads_out = reconstruction.narrow(1, 0, self.nsamples)
        kmers_out = reconstruction.narrow(1, self.nsamples, self.nkmers)

        # If multiple samples, apply softmax
        # if self.nsamples > 1:
        reads_out = softmax(reads_out, dim=1)

        kmers_out = softmax(kmers_out, dim=1)

        return reads_out, kmers_out
    
    def forward(    
        self, *args, mode='train'
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if mode == 'train':
            reads = args[0]
            kmers = args[1]
            log_Rc_read = torch.log(args[2])
            log_Rc_read_xp = torch.log(args[3])
            log_length_x = torch.log(args[4])
            log_length_xp = torch.log(args[5])
            input_tensor = torch.cat((reads, kmers), 1)
            mu, logvar = self._encode(input_tensor, log_Rc_read, log_length_x)
            latent = self.reparameterize(mu, logvar)
            reads_out, kmers_out = self._decode(latent, log_Rc_read_xp, log_length_xp)

        else:
            reads = args[0]
            kmers = args[1]
            log_Rc_read = torch.log(args[2])
            log_length = torch.log(args[3])
            input_tensor = torch.cat((reads, kmers), 1)
            mu, logvar = self._encode(input_tensor, log_Rc_read, log_length)
            latent = self.reparameterize(mu, logvar)
            reads_out, kmers_out = self._decode(latent, log_Rc_read, log_length)
        
        return reads_out, kmers_out, mu, logvar
    
    
    def calc_loss(
        self,
        reads_in: Tensor,
        reads_out: Tensor,
        kmers_in: Tensor,
        kmers_out: Tensor,
        Rc_xp: Tensor,
        lengths_xp: Tensor,
        mu: Tensor,
        logvar: Tensor,
        # weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        
        """ McDevol multinomial distribution-based loss function """

        readweights = 1 / torch.pow(Rc_xp, 0.2)
        read_loss = - readweights * (reads_in * torch.log(reads_out + 1e-9)).sum(axis=1)
        
        lenweights = 1 / torch.pow(lengths_xp, 0.6)
        kmer_loss = - lenweights * (kmers_in * torch.log(kmers_out + 1e-9)).sum(axis=1)

        kld_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

        reconstruction_loss = read_loss + kmer_loss

        loss = reconstruction_loss + kld_loss

        return loss.mean(), read_loss.mean(), kmer_loss.mean(), reconstruction_loss.mean(), kld_loss.mean()


    def trainepoch(
        self,
        cluster_ids,
        nepochs: int,
        optimizer,
        batchsteps: list[int],
        logfile,
    ):


        batch_size = 256

        for epoch in range(nepochs):

            self.train()

            epoch_loss = 0.0
            epoch_readloss = 0.0
            epoch_kmerloss = 0.0
            epoch_reloss = 0.0
            epoch_kldloss = 0.0


            pair_indices = get_randompairs(cluster_ids)
            dataloader = DataLoader(dataset=pair_indices, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers, pin_memory=self.cuda)

            if epoch in batchsteps:
                batch_size = batch_size * 2
                dataloader = DataLoader(dataset=dataloader.dataset, batch_size= batch_size, shuffle=True, drop_last=True, num_workers=self.num_workers, pin_memory=self.cuda)

            for indices in dataloader:
                
                reads_in = self.read_counts[indices[:,0]]
                kmers_in = self.kmer_counts[indices[:,0]]
                Rc_reads_in = self.Rc_reads[indices[:,0]]
                lengths_in = self.lengths[indices[:,0]]

                reads_xp_in = self.read_counts[indices[:,1]]
                kmers_xp_in = self.kmer_counts[indices[:,1]]
                Rc_reads_xp_in = self.Rc_reads[indices[:,1]]
                lengths_xp_in = self.lengths[indices[:,1]]

                # reads_in.requires_grad = True
                # kmers_in.requires_grad = True
                # Rc_reads_in.requires_grad = True
                # lengths_in.requires_grad = True

                # lengths_pairs = torch.cat((lengths_in.reshape(-1,1), lengths_xp_in.reshape(-1,1)),1)
                # lengths_avg = torch.mean(lengths_pairs,1)  # np.max(lengths,axis=1)# or np.min(lengths,axis=1)
                # weights = torch.log(lengths_avg) - 5.0
                # weights[weights < 2.0] = 2.0
                # weights *= len(weights) / weights.sum(axis=0)

                if self.usecuda:
                    reads_in = reads_in.cuda()
                    kmers_in = kmers_in.cuda()
                    Rc_reads_in = Rc_reads_in.cuda()
                    lengths_in = lengths_xp_in.cuda()
                    reads_xp_in = reads_xp_in.cuda()
                    kmers_xp_in = kmers_xp_in.cuda()
                    Rc_reads_xp_in = Rc_reads_xp_in.cuda()
                    lengths_xp_in = lengths_xp_in.cuda()
                    weights = weights.cuda()

                
                optimizer.zero_grad()
            
                reads_out, kmers_out, mu, logvar = self(reads_in, kmers_in, Rc_reads_in, Rc_reads_xp_in, lengths_in, lengths_xp_in)

                loss, readloss, kmerloss, reloss, kld = self.calc_loss(
                    reads_xp_in, reads_out, kmers_xp_in, kmers_out, Rc_reads_xp_in, lengths_xp_in, mu, logvar #, weights
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.data.item()
                epoch_readloss += readloss.data.item()
                epoch_kmerloss += kmerloss.data.item()
                epoch_reloss += reloss.data.item()
                epoch_kldloss += kld.data.item()
                # print(torch.min(logvar), torch.max(logvar), torch.mean(logvar), "complete after loss")

                if epoch in batchsteps:
                    print(torch.min(logvar), torch.max(logvar), torch.mean(logvar), "complete after loss", epoch, "epoch")
                #     torch.save(mu, "/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/analyze_latent/mu_" + str(epoch) + ".t")
                #     torch.save(logvar, "/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/analyze_latent/logvar_" + str(epoch) + ".t")
            
            print(
                "\tEpoch: {}\tLoss: {:.6f}\tRLoss: {:.7f}\tKLoss: {:.6f}\tKLD: {:.4f}\tBatchsize: {}".format(
                    epoch + 1,
                    epoch_loss / len(dataloader),
                    epoch_readloss / len(dataloader),
                    epoch_kmerloss / len(dataloader),
                    epoch_kldloss / len(dataloader),
                    dataloader.batch_size,
                ))
            
        if logfile is not None:
            print(
                "\tEpoch: {}\tLoss: {:.6f}\tRLoss: {:.7f}\tKLoss: {:.6f}\tKLD: {:.4f}\tBatchsize: {}".format(
                    epoch + 1,
                    epoch_loss / len(dataloader),
                    epoch_readloss / len(dataloader),
                    epoch_kmerloss / len(dataloader),
                    epoch_reloss / len(dataloader),
                    epoch_kldloss / len(dataloader),
                    dataloader.batch_size,
                ),
                file=logfile,
            )

            logfile.flush()

        self.eval()

        return batch_size


    def trainmodel(
        self,
        nepochs: int = 300,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [25, 75, 150, 200], #[10, 20, 30, 45],
        logfile: Optional[IO[str]] = None,
        modelfile: Union[None, str, Path, IO[bytes]] = None,
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
            batchsteps_set = set(batchsteps)

        if logfile is not None:
            print("\tNetwork properties:", file=logfile)
            print("\tCUDA:", self.usecuda, file=logfile)
            print("\tAlpha:", self.alpha, file=logfile)
            print("\tBeta:", self.beta, file=logfile)
            print("\tDropout:", self.dropout, file=logfile)
            print("\tN hidden:", ", ".join(map(str, self.nhiddens)), file=logfile)
            print("\tN latent:", self.nlatent, file=logfile)
            print("\n\tTraining properties:", file=logfile)
            print("\tN epochs:", nepochs, file=logfile)
            batchsteps_string = (
                ", ".join(map(str, sorted(batchsteps_set)))
                if batchsteps_set
                else "None"
            )
            print("\tBatchsteps:", batchsteps_string, file=logfile)
            print("\tLearning rate:", lrate, file=logfile)
            print("\tN sequences:", self.ncontigs, file=logfile)
            print("\tN samples:", self.nsamples, file=logfile, end="\n\n")

        optimizer = Adam(self.parameters(), lr=lrate)
        # cluster_ids = pd.read_csv('/scratch/users/yazhini.a01/vamb_runs/cluster_assigned', header=None).to_numpy().ravel()
        cluster_ids = pd.read_csv('/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/cluster_assigned_q10q9', header=None).to_numpy().ravel()

        self.trainepoch(
            cluster_ids, nepochs, optimizer, sorted(batchsteps_set), logfile
        )

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None
            
    # def encode_latent(self):
    def encode(self):
        """Encode a data loader to a latent representation with VAE
        Input: data_loader: As generated by train_vae
        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        dataset = TensorDataset(self.read_counts, self.kmer_counts, self.Rc_reads, self.lengths)
        encode_data = DataLoader(dataset=dataset, batch_size=2048, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=self.cuda)

        latent_mu = np.empty((self.ncontigs, self.nlatent), dtype=np.float32)
        latent_logvar = np.empty((self.ncontigs, self.nlatent), dtype=np.float32)

        row = 0
        with torch.no_grad():
            for read_counts, kmer_counts, Rc_reads, lengths in encode_data:
                # Move input to GPU if requested
                if self.usecuda:
                    read_counts = read_counts.cuda()
                    kmer_counts = kmer_counts.cuda()
                    Rc_reads = Rc_reads.cuda()
                    lengths = lengths.cuda()

                # Evaluate
                _, _, mu, logvar = self(read_counts, kmer_counts, Rc_reads, lengths, mode='encode')

                if self.usecuda:
                    mu = mu.cpu()
                    logvar = logvar.cpu()

                latent_mu[row : row + len(mu)] = mu
                latent_logvar[row : row + len(mu)] = logvar
                row += len(mu)

        assert row == self.ncontigs
        return latent_mu, latent_logvar
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
import numpy as np
import torch

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

def zscore(
    array: np.ndarray, axis: Optional[int] = None, inplace: bool = False
) -> np.ndarray:
    """Calculates zscore for an array.

    Inputs:
        array: Numpy array to be normalized
        axis: Axis to operate across [None = whole array]
        inplace: Do not create new array, change input array [False]

    Output:
        If inplace is True: Input numpy array
        else: New normalized Numpy-array
    """

    if axis is not None and (axis >= array.ndim or axis < 0):
        raise np.AxisError(str(axis))

    if inplace and not np.issubdtype(array.dtype, np.floating):
        raise TypeError("Cannot convert a non-float array to zscores")

    mean = array.mean(axis=axis)
    std = array.std(axis=axis)

    if axis is None:
        if std == 0:
            std = 1  # prevent divide by zero

    else:
        std[std == 0.0] = 1  # prevent divide by zero
        shape = tuple(dim if ax != axis else 1 for ax, dim in enumerate(array.shape))
        mean.shape, std.shape = shape, shape

    if inplace:
        array -= mean
        array /= std
        return array
    
    else:
        return (array - mean) / std


def get_dataloader_training(
    rpkm: np.ndarray,
    tnf: np.ndarray,
    Rc_reads: np.ndarray, ## my edit to include x_prime
    lengths: np.ndarray,
    cluster_ids: np.ndarray,
    batchsize: int = 256,
    destroy: bool = False,
    cuda: bool = False
    ):

    pair_indices = get_randompairs(cluster_ids)

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
    print(sample_depths_sum)
    if np.any(sample_depths_sum == 0):
        raise ValueError(
            "One or more samples have zero depth in all sequences, so cannot be depth normalized"
        )
    rpkm *= 1_000_000 / sample_depths_sum
    print(rpkm.sum(axis=1))
    # If multiple samples, also include nonzero depth as requirement for accept
    # of sequences
    mask = tnf.sum(axis=1) != 0
    
    depthssum = None
    
    if rpkm.shape[1] > 1:
        depthssum = rpkm.sum(axis=1)
        mask &= depthssum != 0
        depthssum = depthssum[mask]
        assert isinstance(depthssum, np.ndarray)

    if mask.sum() < batchsize:
        raise ValueError(
            "Fewer sequences left after filtering than the batch size. "
            + "This probably means you try to run on a too small dataset (below ~10k sequences), "
            + "or that nearly all sequences were filtered away. Check the log file, "
            + "and verify BAM file content is sensible."
        )

    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        assert depthssum is not None  # we set it so just above
        rpkm /= depthssum.reshape((-1, 1))
    else:
        zscore(rpkm, axis=0, inplace=True)

        
    # ## START my edit to include x_prime    
    # if rpkm_xp.shape[1] > 1:
    #     assert depthssum_xp is not None  # we set it so just above
    #     rpkm_xp /= depthssum_xp.reshape((-1, 1))
    # else:
    #     zscore(rpkm_xp, axis=0, inplace=True)
    # ## END my edit to include x_prime
        
    # Normalize TNF
    # zscore(tnf, axis=0, inplace=True)

    # zscore(tnf_xp, axis=0, inplace=True) ## my edit to include x_prime
    sample_depths_sum = tnf.sum(axis=0)
    tnf *= 1_000_000 / sample_depths_sum

    # # Create weights
    lengths_pairs = np.concatenate((lengths[pair_indices[:,0]].reshape(-1,1), lengths[pair_indices[:,1]].reshape(-1,1)),axis=1) ## my edit to include x_prime
    lengths_avg = np.average(lengths_pairs,axis=1)  # np.max(lengths,axis=1)# or np.min(lengths,axis=1) ## my edit to include x_prime
    # lengths_avg = (lengths_avg[mask]).astype(np.float32)
    weights = np.log(lengths_avg).astype(np.float32) - 5.0
    weights[weights < 2.0] = 2.0
    weights *= len(weights) / weights.sum(axis=0)
    weights.shape = (len(weights), 1)

    ### Create final tensors and dataloader ###

    rpkm_x = rpkm[pair_indices[:,0]]
    rpkm_xp = rpkm[pair_indices[:,1]]
    tnf_x = tnf[pair_indices[:,0]]
    tnf_xp = tnf[pair_indices[:,1]]
    Rc_x = Rc_reads[pair_indices[:,0]]
    Rc_xp = Rc_reads[pair_indices[:,1]]
    lengths_x = lengths[pair_indices[:,0]]
    lengths_xp = lengths[pair_indices[:,1]]
    
    # ## START my edit to include x_prime 
    depths_tensor = torch.from_numpy(rpkm_x)
    tnf_tensor = torch.from_numpy(tnf_x)
    weightstensor = torch.from_numpy(weights)
    depths_xp_tensor = torch.from_numpy(rpkm_xp)
    tnf_xp_tensor = torch.from_numpy(tnf_xp)
    Rc_reads_tensor = torch.from_numpy(Rc_x)
    Rc_reads_xp_tensor = torch.from_numpy(Rc_xp)
    lengths_x_tensor = torch.from_numpy(lengths_x)
    lengths_xp_tensor = torch.from_numpy(lengths_xp)
    # ## END my edit to include x_prime
    
    n_workers = 4 if cuda else 1
    dataset = TensorDataset(depths_tensor, tnf_tensor, Rc_reads_tensor, lengths_x_tensor, depths_xp_tensor, tnf_xp_tensor, Rc_reads_xp_tensor, lengths_xp_tensor, weightstensor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=True,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
    )

    return dataloader


def get_dataloader(
    rpkm: np.ndarray,
    tnf: np.ndarray,
    lengths: np.ndarray,
    batchsize: int = 256,
    destroy: bool = False,
    cuda: bool = False,
) -> tuple[DataLoader[tuple[Tensor, Tensor, Tensor]], np.ndarray]:

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
        mask &= depthssum != 0
        depthssum = depthssum[mask]
        assert isinstance(depthssum, np.ndarray)
        
    if mask.sum() < batchsize:
        raise ValueError(
            "Fewer sequences left after filtering than the batch size. "
            + "This probably means you try to run on a too small dataset (below ~10k sequences), "
            + "or that nearly all sequences were filtered away. Check the log file, "
            + "and verify BAM file content is sensible."
        )


    # If multiple samples, normalize to sum to 1, else zscore normalize
    if rpkm.shape[1] > 1:
        assert depthssum is not None  # we set it so just above
        rpkm /= depthssum.reshape((-1, 1))
    else:
        zscore(rpkm, axis=0, inplace=True)

        
    # Normalize TNF
    # zscore(tnf, axis=0, inplace=True)
    sample_depths_sum = tnf.sum(axis=0)
    tnf *= 1_000_000 / sample_depths_sum
    
    # Create weights
    lengths = (lengths[mask]).astype(np.float32)
    weights = np.log(lengths).astype(np.float32) - 5.0
    weights[weights < 2.0] = 2.0
    weights *= len(weights) / weights.sum(axis=0)
    weights.shape = (len(weights), 1)

    ### Create final tensors and dataloader ###
    depthstensor = torch.from_numpy(rpkm)
    tnftensor = torch.from_numpy(tnf)
    weightstensor = torch.from_numpy(weights)
    
    n_workers = 4 if cuda else 1
    dataset = TensorDataset(depthstensor, tnftensor, weightstensor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        drop_last=True,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=cuda,
    )

    return dataloader
class mcdevol_VAE(nn.Module):
    """Variational autoencoder, subclass of torch.nn.Module.

    Instantiate with:
        nsamples: Number of samples in abundance matrix
        nhiddens: list of n_neurons in the hidden layers [None=Auto]
        nlatent: Number of neurons in the latent layer [32]
        alpha: Approximate starting TNF/(CE+TNF) ratio in loss. [None = Auto]
        beta: Multiply KLD by the inverse of this value [200]
        dropout: Probability of dropout on forward pass [0.2]
        cuda: Use CUDA (GPU accelerated training) [False]

    vae.trainmodel(dataloader, nepochs batchsteps, lrate, logfile, modelfile)
        Trains the model, returning None

    vae.encode(self, data_loader):
        Encodes the data in the data loader and returns the encoded matrix.

    If alpha or dropout is None and there is only one sample, they are set to
    0.99 and 0.0, respectively
    """

    def __init__(
        self,
        nsamples: int,
        nhiddens: Optional[list[int]] = None,
        nlatent: int = 32,
        alpha: Optional[float] = None,
        beta: float = 200.0,
        dropout: Optional[float] = 0.2,
        cuda: bool = False,
        seed: int = 0,
    ):
        if nlatent < 1:
            raise ValueError(f"Minimum 1 latent neuron, not {nlatent}")

        if nsamples < 1:
            raise ValueError(f"nsamples must be > 0, not {nsamples}")

        # If only 1 sample, we weigh alpha and nhiddens differently
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

        super(mcdevol_VAE, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.nsamples = nsamples
        self.ntnf = 256 #103
        self.alpha = alpha
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout

        # Initialize lists for holding hidden layers
        # self.encoderlayers = nn.ModuleList()
        # self.encodernorms = nn.ModuleList()
        # self.decoderlayers = nn.ModuleList()
        # self.decodernorms = nn.ModuleList()

        # encoding with two fully connected linear layers
        self.encoder_fc1 = nn.Linear(self.nsamples + self.ntnf + 2, self.nhiddens[0])
        self.encoder_fc2 = nn.Linear(self.nhiddens[0] + 2, self.nhiddens[0])      
        self.encodernorm = nn.BatchNorm1d(self.nhiddens[0]) 

        # Latent layers
        self.mu = nn.Linear(self.nhiddens[0], self.nlatent)
        self.logsigma = nn.Linear(self.nhiddens[0], self.nlatent)


        # decoding with two fully connected linear layers
        self.decoder_fc1 = nn.Linear(self.nlatent + 2, self.nhiddens[0])
        self.decoder_fc2 = nn.Linear(self.nhiddens[0] + 2, self.nhiddens[0])
        self.decodernorm = nn.BatchNorm1d(self.nhiddens[0]) 
            
        # Reconstruction (output) layer
        self.outputlayer = nn.Linear(self.nhiddens[0], self.nsamples + self.ntnf)

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.dropoutlayer = nn.Dropout(p=self.dropout)

        if cuda:
            self.cuda()

    def encode(self, input_tensor: Tensor, Rc_tensor: Tensor, lengths_tensor: Tensor) -> tuple[Tensor, Tensor]:

        tensor = self.encodernorm(self.dropoutlayer(self.relu(self.encoder_fc1(torch.cat((input_tensor, Rc_tensor[:,None], lengths_tensor[:,None]), 1)))))
        tensor = self.encodernorm(self.dropoutlayer(self.relu(self.encoder_fc2(torch.cat((tensor, Rc_tensor[:,None], lengths_tensor[:,None]), 1)))))
        mu = self.mu(tensor)
        logsigma = self.softplus(self.logsigma(tensor))

        return mu, logsigma

    def reparameterize(self, mu: Tensor, logsigma: Tensor) -> Tensor:
        
        epsilon = torch.randn(mu.size(0), mu.size(1))
        if self.usecuda:
            epsilon = epsilon.cuda()

        epsilon.requires_grad = True
        latent = mu + epsilon * torch.exp(logsigma / 2)

        return latent

    def decode(self, latent_tensor: Tensor, Rc_xp_tensor: Tensor, lengths_xp_tensor: Tensor) -> tuple[Tensor, Tensor]:
        
        tensor = self.encodernorm(self.dropoutlayer(self.relu(self.decoder_fc1(torch.cat((latent_tensor, Rc_xp_tensor[:,None], lengths_xp_tensor[:,None]), 1)))))
        tensor = self.encodernorm(self.dropoutlayer(self.relu(self.decoder_fc2(torch.cat((tensor, Rc_xp_tensor[:,None], lengths_xp_tensor[:,None]), 1)))))
        reconstruction = self.outputlayer(tensor)

        # Decompose reconstruction to depths and tnf signal
        reads_out = reconstruction.narrow(1, 0, self.nsamples)
        kmers_out = reconstruction.narrow(1, self.nsamples, self.ntnf)

        # If multiple samples, apply softmax
        if self.nsamples > 1:
            reads_out = softmax(reads_out, dim=1)

        kmers_out = softmax(kmers_out, dim=1)

        return reads_out, kmers_out

    def forward(
        self, depths: Tensor, tnf: Tensor, Rc_read: Tensor, length_x: Tensor, Rc_read_xp: Tensor, length_xp: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        input_tensor = torch.cat((depths, tnf), 1)
        mu, logsigma = self.encode(input_tensor, Rc_read, length_x)
        latent = self.reparameterize(mu, logsigma)
        depths_out, tnf_out = self.decode(latent, Rc_read_xp, length_xp)

        return depths_out, tnf_out, mu, logsigma

    def calc_loss(
        self,
        reads_in: Tensor,
        reads_out: Tensor,
        kmers_in: Tensor,
        kmers_out: Tensor,
        mu: Tensor,
        logsigma: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        
        loss_read = - (reads_in * torch.log(reads_out + 1e-9)).sum(axis=1)
        loss_kmer = - (kmers_in * torch.log(kmers_out + 1e-9)).sum(axis=1)
        kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1)
        kld_weight = 1 / (self.nlatent * self.beta)
        reconstruction_loss = loss_read + loss_kmer
        kld_loss = kld * kld_weight
        loss = (reconstruction_loss + kld_loss) * weights
        return loss.mean(), loss_read.mean(), loss_kmer.mean(), kld.mean()

    def trainepoch(
        self,
        data_loader: DataLoader,
        epoch: int,
        optimizer,
        batchsteps: list[int],
        logfile,
    ) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]: ## my edit to include x_prime
        self.train()

        epoch_loss = 0.0
        epoch_kldloss = 0.0
        epoch_sseloss = 0.0
        epoch_celoss = 0.0

        if epoch in batchsteps:
            data_loader = set_batchsize(data_loader, data_loader.batch_size * 2)

        for depths_in, tnf_in, Rc_read, length, depths_xp_in, tnf_xp_in, Rc_read_xp, length_xp, weights in data_loader: ## my edit to include x_prime
            
            # depths_in.requires_grad = True
            # tnf_in.requires_grad = True

            # if self.usecuda:
            #     depths_in = depths_in.cuda()
            #     tnf_in = tnf_in.cuda()
            #     weights = weights.cuda()

            optimizer.zero_grad()

            depths_out, tnf_out, mu, logsigma = self(depths_in, tnf_in, Rc_read, length, Rc_read_xp, length_xp)
            
            loss, ce, sse, kld = self.calc_loss(
                depths_xp_in, depths_out, tnf_xp_in, tnf_out, mu, logsigma, weights
            )

            loss.backward()
            optimizer.step()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_sseloss += sse.data.item()
            epoch_celoss += ce.data.item()

        print(
                "\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}".format(
                    epoch + 1,
                    epoch_loss / len(data_loader),
                    epoch_celoss / len(data_loader),
                    epoch_sseloss / len(data_loader),
                    epoch_kldloss / len(data_loader),
                    data_loader.batch_size,
        ))
        if logfile is not None:
            print(
                "\tEpoch: {}\tLoss: {:.6f}\tCE: {:.7f}\tSSE: {:.6f}\tKLD: {:.4f}\tBatchsize: {}".format(
                    epoch + 1,
                    epoch_loss / len(data_loader),
                    epoch_celoss / len(data_loader),
                    epoch_sseloss / len(data_loader),
                    epoch_kldloss / len(data_loader),
                    data_loader.batch_size,
                ),
                file=logfile,
            )

            logfile.flush()

        self.eval()
        return data_loader

    def encode_latent(self, data_loader):
        """Encode a data loader to a latent representation with VAE

        Input: data_loader: As generated by train_vae

        Output: A (n_contigs x n_latent) Numpy array of latent repr.
        """

        self.eval()

        new_data_loader = set_batchsize(
            data_loader, data_loader.batch_size, encode=True
        )

        depths_array, _ , _ = data_loader.dataset.tensors ## my edit to include x_prime
        length = len(depths_array)

        # We make a Numpy array instead of a Torch array because, if we create
        # a Torch array, then convert it to Numpy, Numpy will believe it doesn't
        # own the memory block, and array resizes will not be permitted.
        latent_mu = np.empty((length, self.nlatent), dtype=np.float32)
        latent_logsigma = np.empty((length, self.nlatent), dtype=np.float32)

        row = 0
        with torch.no_grad():
            for depths, tnf, _ in new_data_loader: ## my edit to include x_prime
                # Move input to GPU if requested
                if self.usecuda:
                    depths = depths.cuda()
                    tnf = tnf.cuda()

                # Evaluate
                _, _, mu, logsigma = self(depths, tnf)

                if self.usecuda:
                    mu = mu.cpu()
                    logsigma = logsigma.cpu()

                latent_mu[row : row + len(mu)] = mu
                latent_logsigma[row : row + len(mu)] = logsigma
                row += len(mu)

        assert row == length
        return latent_mu, latent_logsigma

    def save(self, filehandle):
        """Saves the VAE to a path or binary opened file. Load with VAE.load

        Input: Path or binary opened filehandle
        Output: None
        """
        state = {
            "nsamples": self.nsamples,
            "alpha": self.alpha,
            "beta": self.beta,
            "dropout": self.dropout,
            "nhiddens": self.nhiddens,
            "nlatent": self.nlatent,
            "state": self.state_dict(),
        }

        torch.save(state, filehandle)

    @classmethod
    def load(
        cls, path: Union[IO[bytes], str], cuda: bool = False, evaluate: bool = True
    ):
        """Instantiates a VAE from a model file.

        Inputs:
            path: Path to model file as created by functions VAE.save or
                  VAE.trainmodel.
            cuda: If network should work on GPU [False]
            evaluate: Return network in evaluation mode [True]

        Output: VAE with weights and parameters matching the saved network.
        """

        # Forcably load to CPU even if model was saves as GPU model
        dictionary = torch.load(path, map_location=lambda storage, loc: storage)

        nsamples = dictionary["nsamples"]
        alpha = dictionary["alpha"]
        beta = dictionary["beta"]
        dropout = dictionary["dropout"]
        nhiddens = dictionary["nhiddens"]
        nlatent = dictionary["nlatent"]
        state = dictionary["state"]

        vae = cls(nsamples, nhiddens, nlatent, alpha, beta, dropout, cuda)
        vae.load_state_dict(state)

        if cuda:
            vae.cuda()

        if evaluate:
            vae.eval()

        return vae

    def trainmodel(
        self,
        dataloader,
        nepochs: int = 5,
        lrate: float = 1e-3,
        batchsteps: Optional[list[int]] = [1, 2, 3, 4],#[25, 75, 150, 300],
        logfile: Optional[IO[str]] = None,
        modelfile: Union[None, str, Path, IO[bytes]] = None,
    ):
        """Train the autoencoder from depths array and tnf array.

        Inputs:
            dataloader: DataLoader made by make_dataloader
            nepochs: Train for this many epochs before encoding [500]
            lrate: Starting learning rate for the optimizer [0.001]
            batchsteps: None or double batchsize at these epochs [25, 75, 150, 300]
            logfile: Print status updates to this file if not None [None]
            modelfile: Save models to this file if not None [None]

        Output: None
        """

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
            last_batchsize = dataloader.batch_size * 2 ** len(batchsteps)
            if len(dataloader.dataset) < last_batchsize:  # type: ignore
                raise ValueError(
                    f"Last batch size of {last_batchsize} exceeds dataset length "
                    f"of {len(dataloader.dataset)}. "  # type: ignore
                    "This means you have too few contigs left after filtering to train. "
                    "It is not adviced to run Vamb with fewer than 10,000 sequences "
                    "after filtering. "
                    "Please check the Vamb log file to see where the sequences were "
                    "filtered away, and verify BAM files has sensible content."
                )
            batchsteps_set = set(batchsteps)

        # Get number of features
        # Following line is un-inferrable due to typing problems with DataLoader
        ncontigs, nsamples = dataloader.dataset.tensors[0].shape  # type: ignore
        optimizer = Adam(self.parameters(), lr=lrate)

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
            print("\tStarting batch size:", dataloader.batch_size, file=logfile)
            batchsteps_string = (
                ", ".join(map(str, sorted(batchsteps_set)))
                if batchsteps_set
                else "None"
            )
            print("\tBatchsteps:", batchsteps_string, file=logfile)
            print("\tLearning rate:", lrate, file=logfile)
            print("\tN sequences:", ncontigs, file=logfile)
            print("\tN samples:", nsamples, file=logfile, end="\n\n")

        # Train
        for epoch in range(nepochs):
            dataloader = self.trainepoch(
                dataloader, epoch, optimizer, sorted(batchsteps_set), logfile
            )

        # Save weights - Lord forgive me, for I have sinned when catching all exceptions
        if modelfile is not None:
            try:
                self.save(modelfile)
            except:
                pass

        return None
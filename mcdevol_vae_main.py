#!/usr/bin/env python

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Optional, IO
import matplotlib.pyplot as plt
import torch

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import src.mcdevol_variational_autoencoder as mcdevol_vae
import src.mcdevol_autoencoder as mcdevol_ae
# import src.mcdevol_autoencoder_trainingtesting as mcdevol_ae
import src.mcdevol_autoencoder_augmented_both as mcdevol_ae
#import src.nmf as nmf_call
# import src.nmf_connected_components as nmf_cc

# sys.path.insert(0, '/big/software/vamb')

# import vamb.cluster as vamb_clusters
# import vamb._vambtools


def call_ae(args):

    if not args.autoencoder:
        vae = mcdevol_vae.mcdevol_VAE(args)
        vae.trainmodel(logfile=args.logfile)
        vae.testmodel(logfile=args.logfile)
        latent_mu, latent_logvar = vae.getlatent(logfile=args.logfile)
        # num_processes = 12
        # vae.share_memory()
        # p = torch_mp.Process(target=vae.trainmodel(logfile=args.logfile), args=(vae,))
        # p.start()

        # p.join()

        np.save(args.outdir + f'/latent_mu_{args.logvar}_{args.nlatent}_{args.flag}_{args.kmerweight}.npy', latent_mu)
        np.save(args.outdir + f'/latent_var_{args.logvar}_{args.nlatent}_{args.flag}_{args.kmerweight}.npy', np.exp(latent_logvar))


    else:
        ae = mcdevol_ae.mcdevol_AE(args)
        ae.trainmodel(logfile=args.logfile)
        ae.testmodel(logfile=args.logfile)
        latent = ae.getlatent(logfile=args.logfile)
        np.save(args.outdir + f'/latent_mu.npy', latent)

        # ae = mcdevol_ae.mcdevol_AE(args)
        # ae.load_state_dict(torch.load(args.outdir, 'autoencoder_model.pth', map_location=torch.device('cpu')))
        # print("getting latent space")
        # latent_of_bins = ae.latent_ofbins(args.binreads, args.binkmers, args.binlength, logfile = args.logfile)
        # np.save(args.outdir + f'/latent_ofbins.npy', latent_of_bins)

    return None


# def nmf_call(args, read_counts, contig_length, clusters, numclust_incomponents, an, alpha):
#     clusters = pd.read_csv(args.outdir + '/cluster_cosine.csv', sep='\t', header=None)
#     nmf_cc(read_counts, contig_length, clusters, numclust_incomponents, an, alpha, small_nmfassign_flag=1, mode = 0)

def main():

    doc = f""" variational autoencoder for metagenome binning """

    start = time.time()
    parser = argparse.ArgumentParser(
        prog="mcdevol",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s --reads read_counts --kmers kmer_counts --length --outdir [options]",
        add_help=False,
    )

    parser.add_argument("--reads", type=str, help="directory that contains all alignment files in bam format", required=True)
    parser.add_argument("--kmers", type=str, help="contig sequence file in fasta format (or zip)", required=True)
    parser.add_argument("--codons", type=str, help="contig counts", required=True)
    parser.add_argument("--length", type=str, help="length of contigs in bp", required=True)
    parser.add_argument("--names", type=str, help="ids of contigs", required=True)
    parser.add_argument("--outdir", type=str, help="output directory", required=True)
    parser.add_argument("--flag", type=str, help="for read counts or kmer counts or both counts vae", default='both')
    parser.add_argument("--nlatent", type=int, help="number of latent space")
    parser.add_argument("--kmerweight", type=float, help="set kmerweight between 0.1 to 1")
    parser.add_argument("--codonweight", type=float, help="set codonweight between 0.1 to 1")
    parser.add_argument("--logvar", type=int, help="logvar status 0: learn for all nlatent space variance, 1: learn one variance, 2: fix it to 1")
    parser.add_argument("--autoencoder", help="run autoencoder instead of variational autoencoder", action="store_true")
    parser.add_argument("--zscore", help="input normalization type, [False]: x/Rc, [True]: zscore", action="store_true")
    parser.add_argument("--cuda", help="use GPU to train & cluster [False]", action="store_true")

    args = parser.parse_args()

    args.reads = np.load(args.reads, allow_pickle=True)['arr_0']
    args.kmers = np.load(args.kmers, allow_pickle=True)['arr_0'] # preprossed (divide by 2, correction for repeat region)
    args.length = np.load(args.length, allow_pickle=True)['arr_0']
    args.names = np.load(args.names, allow_pickle=True)['arr_0']
    args.codons = np.load(args.codons, allow_pickle=True)['arr_0']

    args.kmers_left = np.load(args.outdir + 'mcdevol_kmercounts_left.npz', allow_pickle=True)['arr_0']
    args.kmers_right = np.load(args.outdir + 'mcdevol_kmercounts_right.npz', allow_pickle=True)['arr_0']
   
    # args.binreads = np.load('/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/pseudocounts.npy')
    # args.binkmers = np.load('/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/summedkmercounts.npy')
    # args.binlength = np.load('/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/summedlength.npy')

    """ remove stop codon column """
    # args.codons = np.delete(args.codons,[16,18], axis=1)

    print(args.cuda, 'cuda option', args.flag, args.nlatent, args.codonweight)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    args.logfile = open(args.outdir + f'/log_vae.txt', 'w')

    call_ae(args)

    # bins = nmf_call.nmf(args)
    # np.savetxt(args.outdir + '/' +"bin_assignments_1163", np.stack((args.names[bins[0]], bins[1])).T, fmt='%s,%d')
    print(f"metagenome binning is completed in {time.time() - start} seconds", file=args.logfile)

    args.logfile.close()
    
    return None

if __name__ == "__main__" :
    main()

#!/usr/bin/env python

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Optional, IO
import matplotlib.pyplot as plt

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import src.mcdevol_autoencoder_nmfbins as mcdevol_ae
import torch
import torch.autograd.profiler as profiler

def call_ae(args):


    ae = mcdevol_ae.mcdevol_AE(args)
    ae.trainmodel(logfile=args.logfile)
    ae.testmodel(logfile=args.logfile)
    # print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=50))
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
        usage="%(prog)s --reads_matrix pseudocounts --kmers_matrix summed_kmercounts --kmers_all kmer_counts --contiglists contig_list --Zbcmatrix Z_matrix --outdir path",
        add_help=False,
    )


    parser.add_argument("--reads_matrix", type=str, help="directory that contains all alignment files in bam format", required=True)
    parser.add_argument("--kmers_matrix", type=str, help="matrix", required=True)
    parser.add_argument("--kmers_all", type=str, help="matrix", required=True)
    parser.add_argument("--contiglists", type=str, help="contig counts", required=True)
    parser.add_argument("--Zbcmatrix", type=str, help='Zbc vector per bin obtained from NMF', required=True)
    parser.add_argument("--outdir", type=str, help="output directory", required=True)
    parser.add_argument("--flag", type=str, help="for read counts or kmer counts or both counts vae", default='both')
    parser.add_argument("--nlatent", type=int, help="number of latent space")
    parser.add_argument("--kmerweight", type=float, help="set kmerweight between 0.1 to 1")
    parser.add_argument("--logvar", type=int, help="logvar status 0: learn for all nlatent space variance, 1: learn one variance, 2: fix it to 1")
    parser.add_argument("--cuda", help="use GPU to train & cluster [False]", action="store_true")


    args = parser.parse_args()

    args.reads = np.load(args.reads_matrix, allow_pickle=True)
    args.kmers = np.load(args.kmers_matrix, allow_pickle=True) # preprossed (divide by 2, correction for repeat region)
    args.kmers_all = np.load(args.kmers_all, allow_pickle=True)['arr_0']
    args.contigids_inbins = np.load(args.contiglists, allow_pickle=True)
    args.Zbcmatrix = np.load(args.Zbcmatrix, allow_pickle=True)

    print(args.cuda, 'cuda option', args.flag, args.nlatent)
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    args.logfile = open(args.outdir + f'/log_autoencoder_bins.txt', 'w')

    call_ae(args)

    # bins = nmf_call.nmf(args)
    # np.savetxt(args.outdir + '/' +"bin_assignments_1163", np.stack((args.names[bins[0]], bins[1])).T, fmt='%s,%d')
    print(f"metagenome binning is completed in {time.time() - start} seconds", file=args.logfile)

    args.logfile.close()
    
    return None

if __name__ == "__main__" :
    main()

#!/usr/bin/env python
""" perform augmentation-agumented vae """


import os
import sys
import time
import argparse
import numpy as np
import torch
# import src.iterative_augmentedautoencoder as mcdevol_vae
# import src.vae_standard as mcdevol_vae
import src.vae_byol as mcdevol_vae
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)


def call_ae(args):
    """ call function to perform autoencoder training """
    ae = mcdevol_vae.McdevolVAE(args)
    ae.trainmodel(logfile=args.logfile)
    ae.testmodel(logfile=args.logfile)
    latent = ae.getlatent(args.logfile)
    np.save(args.outdir + '/latent_mu.npy', latent)


    # with links retrain it

    # pretrained_weights = torch.load(args.outdir + '/autoencoder_trainedmodel.pth', map_location=torch.device('cpu'))
    # ae.load_state_dict(pretrained_weights)
    # latent = np.load(args.outdir + '/latent_mu.npy', allow_pickle=True)
    # _, _, neighbors_list, neighbors_1, neighbors_2 = neighbor_search(latent, args.length)
    # ae.train_by_readlinks(pair_links, args.logfile)
    # latent_i1 = ae.getlatent(True, args.logfile)
    # np.save(args.outdir + '/latent_i1.npy', latent_i1)
    # ae.iterative_training(neighbors_list, neighbors_1, neighbors_2, args.logfile)
    # # pretrained_weights = torch.load(args.outdir + 'autoencoder_trainedmodel_i1.pth', map_location=torch.device('cpu'))
    # # ae.load_state_dict(pretrained_weights)

    return None

def main() -> None:

    """ variational autoencoder for metagenome binning """
    start = time.time()
    parser = argparse.ArgumentParser(
        prog="mcdevol",
        description="variational autoencoder for metagenome binning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s \
        --reads read_counts --kmers kmer_counts --kmers_augleft --kmers_augright --length --names --outdir [options]",
        add_help=False,
    )

    parser.add_argument("--reads", type=str, \
        help="directory that contains all alignment files in bam format", required=True)
    parser.add_argument("--kmers", type=str, \
        help="contig sequence file in fasta format (or zip)", required=True)
    parser.add_argument("--kmers_augleft", type=str, \
        help="provide augmented kmer counts (left)", required=True)
    parser.add_argument("--kmers_augright", type=str, \
        help="provide augmented kmer counts (right)", required=True)
    # parser.add_argument("--pair_links", type=str, \
    #     help="provide augmented pair links", required=True)
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
    try:
        if not os.path.exists(args.outdir):
            print('create output folder')
            os.makedirs(args.outdir)
    except Exception as e:
        print(f'output directory already exist. Using it {e}')


    args.logfile = open(args.outdir + '/log_vae.txt', 'w+', encoding='utf-8')

    call_ae(args)

    print(f"metagenome binning is completed in {time.time() - start} seconds", file=args.logfile)

    args.logfile.close()


if __name__ == "__main__" :
    main()

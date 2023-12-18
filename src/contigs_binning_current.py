#!/usr/bin/env python

import os, sys
parent_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_path)

import time
import subprocess
import gc
import numpy as np
import pandas as pd
import src.optimize_parameters as opt
from multiprocessing.pool import Pool
from src.density_based_clustering import cluster_by_centroids
from src.density_based_clustering import cluster_by_connecting_centroids
from util.bam2counts import obtain_readcounts
from memory_profiler import profile
# from src.contigs_clustering import cluster_by_connecting_centroids as min_linkage
import src.iterative_augmentedautoencoder as mcdevol_ae
from datetime import datetime

def write_log(message, logfile):
    print(message, file=logfile)
    logfile.flush()


def optimize_prior_fortrimers(kmer_counts, Rc_kmers, trimercountsper_nt):

    with Pool(ncpu) as pool:
        awa_values = pool.starmap(opt.obtain_optimized_alphas, \
                                [(np.take(kmer_counts,[c*4, (c*4)+1, (c*4)+2, (c*4)+3], axis=1),\
                                np.take(Rc_kmers, c, axis=1), trimercountsper_nt[c]) for c in range(64)])
    awa_values = np.array(awa_values)
    alpha_values = awa_values.sum(axis=1)

    return alpha_values, awa_values

def call_bam2counts(bam):
    obtain_readcounts(bam[0], bam[1], input_dir, tmp_dir, minlength, sequence_identity)

def calcreadcounts(bamfiles):
    with Pool(ncpu) as pool:
        pool.map(call_bam2counts, bamfiles)


def process_kmers(kmer_counts, GC_fractions, contig_length, total_contigs_source):
    print("processing kmer frequencies")
    kmer_counts = kmer_counts.reshape(total_contigs_source, 256) # convert 1D array to a 2D array with {total_contigs, all 4-mers} shape  
    kmer_counts = (kmer_counts / 2)

    GC_tetramer = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
                            2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 1, 1, 2, 2,
                            1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1,
                            2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2,
                            2, 2, 3, 3, 2, 2, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 1, 1,
                            2, 2, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 3,
                            2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4, 3, 3, 4, 4, 2, 2, 3, 3, 2, 2,
                            3, 3, 3, 3, 4, 4, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3,
                            1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3,
                            4, 4, 3, 3, 4, 4, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 4, 4, 3, 3, 4, 4])
    
    """ process repeat regions """
    fGC_tGC = GC_fractions ** GC_tetramer
    fAT_tAT = (1 - GC_fractions) ** (4 - GC_tetramer)
    ekmer_counts = (np.tile((contig_length - 3)[:, None], (1,256)) * fAT_tAT * fGC_tGC)/16
    zscore_kmer = (kmer_counts - ekmer_counts) / np.sqrt(ekmer_counts)
    zmax = 4
    repeat_kmerinds = np.nonzero(zscore_kmer>zmax)

    
    #### set to expected value plus zmax times square root of expected value  ####
    # kmer_counts[repeat_kmerinds] = ekmer_counts[repeat_kmerinds] + zmax * np.sqrt(ekmer_counts[repeat_kmerinds]) 
    
    #### set to expected value (works better) ####
    kmer_counts[repeat_kmerinds] = ekmer_counts[repeat_kmerinds] # replace by mu

    #### set to zero  (not used) ####
    # repeat_index = np.nonzero((kmer_counts / ekmer_counts)>3) # theory suggested z-score but we use ratio for detecting repeat index 
    # setind_zero = np.vstack((np.repeat(repeat_index[0],4),\
    #     np.repeat(repeat_index[1] // 4  * 4, 4) + np.tile([0, 1, 2, 3], np.shape(repeat_index[1]))))
    # kmer_counts[setind_zero[0],setind_zero[1]] = 0

    return kmer_counts

#@profile
def binning(args):

    global input_dir, tmp_dir, minlength, ncpu, sequence_identity, q_read, q_kmer

    input_dir = args.input
    tmp_dir = args.input + '/'
    minlength = args.minlength
    ncpu = args.ncores
    sequence_identity = args.seq_identity

    q_read = np.exp(-0.4)
    q_kmer = 0.2 # np.exp(-6.0)

    s = time.time()
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    write_log(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.logfile)
    # """ obtain read counts """
    # bamfiles = [f for f in os.listdir(input_dir) if f.endswith('.bam')]
    # flags = np.zeros(len(bamfiles), dtype=int)
    # flags[0] = 1
    # bamfiles = list(map(list,zip(bamfiles,flags)))
    
    # if os.path.isfile(tmp_dir + "selected_contigs"):
    #     subprocess.run(["rm " + tmp_dir + "selected_contigs"], shell=True)

    # if any(File.endswith("_count") for File in os.listdir(tmp_dir)):    
    #     subprocess.run(["rm " + tmp_dir + "*_count"], shell=True)
    
    # calcreadcounts(bamfiles)
    subprocess.run(["cat " + tmp_dir + "*_count > " + tmp_dir + "total_readcount"], shell=True)
    # subprocess.run(["awk '{print $1}' " + tmp_dir + "total_readcount | sort -V -u > " + tmp_dir + "selected_contigs"], shell=True)

    # """ obtain kmer counts """
    print(parent_path, 'parent path')
    # subprocess.run([parent_path + "/util/kmerfreq_augment_usetwoseg " + str(tmp_dir) + "  " +str(args.contigs)], shell=True)

    """ clustering parameters (default) """
    d0 = 1.0
 
    """ load contig read counts """
    contigs = pd.read_csv(tmp_dir + 'selected_contigs', header=None, sep=' ').to_numpy()
    contig_names = contigs[:,0]
    contig_length = contigs[:,1].astype(int)
    fractional_counts = pd.read_csv(tmp_dir + "total_readcount", header=None,sep=' ')
    fractional_counts.columns = ['contig_id','sample_id','counts']
    fractional_counts['contig_id'] = fractional_counts['contig_id'].str.replace('PC','').astype(int)
    read_counts = fractional_counts.pivot_table(index = 'sample_id', columns = 'contig_id', values = 'counts').sort_index(axis=1)
    del(fractional_counts)

    read_counts = read_counts.to_numpy().T
    total_contigs_source = read_counts.shape[0]
    print(np.shape(read_counts), 'dimension of read count matrix')

    
    if len(np.nonzero(read_counts.sum(axis=1)==0)[0]) > 0:
        raise RuntimeError("some contigs have zero total counts across samples")

    long_contigs = np.nonzero(contig_length>=minlength)[0]
    contig_length_filtered = contig_length[long_contigs]
    read_counts = read_counts[long_contigs]


    # """ process high read counts """
    # Rc_reads = read_counts.sum(axis=1)
    # Rn_reads = read_counts.sum(axis=0)
   
    # minimap_rpkm = np.load('/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/vamb_output/abundance.npz', allow_pickle=True)

    # minimap_rpkm = minimap_rpkm['matrix']
    # print(np.max(read_counts), np.min(read_counts))
    # read_counts = (minimap_rpkm * contig_length_filtered.reshape(-1,1)) / 300

    # print(read_counts.shape, np.max(read_counts), np.min(read_counts))
    Rc_reads = read_counts.sum(axis=1)
    Rn_reads = read_counts.sum(axis=0)
    total_contigs, _ = np.shape(read_counts)
    ss = time.time()
    dirichlet_prior = opt.optimize_alpha(read_counts, Rc_reads, Rn_reads)
    print('obtained alpha parameter for read counts', dirichlet_prior, 'in' ,time.time()-ss,'seconds')
    write_log(f'obtained alpha parameter for read counts {dirichlet_prior} in {time.time()-ss} seconds', args.logfile)
    ss = time.time()
    dirichlet_prior_persamples  = dirichlet_prior * Rn_reads / Rn_reads.sum()
    print("dirichlet_prior_persamples", dirichlet_prior_persamples)

    """ load kmer counts """
    kmer_counts = pd.read_csv(tmp_dir + "kmer_counts", header=None)
    kmer_counts = kmer_counts.to_numpy()
    print(kmer_counts.shape, 'kmer shape')
    GC_fractions = pd.read_csv(tmp_dir + "GC_fractionof_contigs", header=None).to_numpy()
    kmer_counts = process_kmers(kmer_counts, GC_fractions, contig_length_filtered, total_contigs)

    # length augmentation
    contig_length_aug = np.copy(contig_length_filtered)
    inds_4k = np.nonzero(contig_length>=4000)[0]
    inds_l4k_g2k = np.nonzero((contig_length<4000)&(contig_length*0.9>=2000))[0]
    inds_l4k_l2k = np.nonzero((contig_length<4000)&(contig_length*0.9<2000))[0]
    np.put(contig_length_aug, inds_4k, contig_length[inds_4k]*0.5)
    np.put(contig_length_aug, inds_l4k_g2k, contig_length[inds_l4k_g2k]*0.9)
    np.put(contig_length_aug, inds_l4k_l2k, contig_length[inds_l4k_l2k])
    #################
    
    # left part
    kmer_counts_left = pd.read_csv(tmp_dir + "kmer_counts_left", header=None)
    kmer_counts_left = kmer_counts_left.to_numpy()
    print(kmer_counts_left.shape, 'left shape')
    GC_fractions_left = pd.read_csv(tmp_dir + "GC_fractionof_contigs_left", header=None).to_numpy()
    kmer_counts_left = process_kmers(kmer_counts_left, GC_fractions_left, contig_length_aug, total_contigs)
    #################
    
    # right part
    kmer_counts_right = pd.read_csv(tmp_dir + "kmer_counts_right", header=None)
    kmer_counts_right = kmer_counts_right.to_numpy()
    print(kmer_counts_right.shape, 'right shape')
    GC_fractions_right = pd.read_csv(tmp_dir + "GC_fractionof_contigs_right", header=None).to_numpy()
    kmer_counts_right = process_kmers(kmer_counts_right, GC_fractions_right, contig_length_aug, total_contigs)
    del(contig_length_aug)
    #################

    """ process high kmer counts """
    kmer_counts = kmer_counts[long_contigs]
    kmer_counts_left = kmer_counts_left[long_contigs]
    kmer_counts_right = kmer_counts_right[long_contigs]

    trimercountsper_nt = kmer_counts.reshape(-1,64,4).sum(axis=0)
    Rc_kmers = kmer_counts.reshape(-1,64,4).sum(axis=2)

    ss = time.time()
    dirichlet_prior_kmers, dirichlet_prior_perkmers = optimize_prior_fortrimers(kmer_counts, Rc_kmers, trimercountsper_nt)
    print('obtained alpha parameters for kmer counts in', time.time()-ss,'seconds')
    write_log(f'obtained alpha parameters for kmer counts in {time.time()-ss} seconds', args.logfile)
    del(trimercountsper_nt)

    contig_names_filtered = contig_names[long_contigs]

    # cluster_parameters = list([read_counts, Rc_reads, contig_length, total_contigs, \
    #                       dirichlet_prior, dirichlet_prior_persamples, kmer_counts, Rc_kmers, \
    #                       dirichlet_prior_kmers, dirichlet_prior_perkmers.flatten(), \
    #                       d0, tmp_dir, q_read, q_kmer])   
                          
    np.savez(tmp_dir+'mcdevol_readcounts.npz', read_counts)
    np.savez(tmp_dir+'mcdevol_kmercounts.npz', kmer_counts)
    np.savez(tmp_dir+'mcdevol_kmercounts_right.npz', kmer_counts_left)
    np.savez(tmp_dir+'mcdevol_kmercounts_left.npz', kmer_counts_right)
    np.savez(tmp_dir+'contigs_2klength.npz', contig_length_filtered)
    np.savez(tmp_dir+'contigs_2knames.npz', contig_names_filtered)

    # mcdevol_ae.mcdevol_AE(args)

    # if args.fasta:
    #     subprocess.run([parent_path + "/util/get_sequence_bybin " + str(tmp_dir) + " mcdevol_clusters " + str(args.contigs) + " " + str(args.output) + " " + str(args.outdir)], shell=True)
  
    print('metagenome binning is completed in', time.time()-s,'seconds')
    write_log(f'metagenome binning is completed in {time.time()-s} seconds', args.logfile)

    gc.collect()
    return 0

if __name__ == "__main__":
    pass
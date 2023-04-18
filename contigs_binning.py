#!/usr/bin/env python

import os
import time
import subprocess
import gc
import numpy as np
import pandas as pd
import optimize_parameters as opt
from multiprocessing.pool import Pool
from contigs_clustering import cluster_by_connecting_centroids
import contigs_clustering_modified as ccm
from nmf_connected_components import nmf_connected_components
import assign_shortcontigs as shortcontigs
import density_based_clustering as dc
import psutil
import util.bam2counts as bc
from datetime import datetime

import cProfile, pstats

def optimize_prior_fortrimers(kmer_counts, Rc_kmers, trimercountsper_nt):

    with Pool(ncpu) as pool:
        awa_values = pool.starmap(opt.obtain_optimized_alphas, \
                                [(np.take(kmer_counts,[c*4, (c*4)+1, (c*4)+2, (c*4)+3], axis=1),\
                                np.take(Rc_kmers, c, axis=1), trimercountsper_nt[c]) for c in range(64)])
    awa_values = np.array(awa_values)
    alpha_values = awa_values.sum(axis=1)

    return alpha_values, awa_values

def call_bam2counts(bam):
    bc.obtain_readcounts(bam[0], bam[1], input_dir, tmp_dir, minlength, sequence_identity)

def calcreadcounts(bamfiles):
    with Pool(ncpu) as pool:
        pool.map(call_bam2counts, bamfiles)

def binning(args):

    global input_dir, tmp_dir, minlength, ncpu, sequence_identity

    input_dir = args.input
    tmp_dir = args.input + '/sequence_identity_98p5/'
    minlength = args.minlength
    ncpu = args.ncores
    sequence_identity = args.seq_identity
    s = time.time()
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    """ obtain read counts """
    bamfiles = [f for f in os.listdir(input_dir) if f.endswith('.bam')]
    flags = np.zeros(len(bamfiles), dtype=int)
    flags[0] = 1
    bamfiles = list(map(list,zip(bamfiles,flags)))
    
    if os.path.isfile(tmp_dir + "selected_contigs"):
        subprocess.run(["rm " + tmp_dir + "selected_contigs"], shell=True)

    if any(File.endswith("_count") for File in os.listdir(tmp_dir)):    
        subprocess.run(["rm " + tmp_dir + "*_count"], shell=True)
    
    calcreadcounts(bamfiles)
    subprocess.run(["cat " + tmp_dir + "*_count > " + tmp_dir + "total_readcount"], shell=True)
    
    


    # """ obtain kmer counts """
    # subprocess.run([os.getcwd() + "/util/kmerfreq " + str(tmp_dir) + "  " +str(args.contigs)], shell=True)

    """ clustering parameters (default) """
    d0 = 1.0
    d1 = d0
    min_shared_contigs = 100
    
    print(tmp_dir, "tmp_dir")

    """ load contig read counts """
    contigs = pd.read_csv(tmp_dir + 'selected_contigs', header=None, sep=' ').to_numpy()
    contig_names = contigs[:,1]
    contig_length = contigs[:,2].astype(int)

    fractional_counts = pd.read_csv(tmp_dir + "total_readcount", header=None,sep=' ')
    read_counts = fractional_counts.pivot_table(index = 1, columns = 0, values = 2)
    del(fractional_counts)

    read_counts = read_counts.to_numpy().T
    total_contigs_source = read_counts.shape[0]
    print(np.shape(read_counts))

    """ remove contigs with no counts """
    # Rc_reads = read_counts.sum(axis=1)
    # read_counts = np.delete(read_counts, np.nonzero(Rc_reads==0)[0], axis=0)
    
    if len(np.nonzero(read_counts.sum(axis=1)==0)[0]) > 0:
        raise RuntimeError("some contigs have zero total counts across samples")

    long_contigs_minlength = 2500
    long_contigs = np.nonzero(contig_length>=long_contigs_minlength)[0]
    read_counts = read_counts[long_contigs]
    total_contigs, n_size = np.shape(read_counts)

    print(np.shape(read_counts))

    """ process high read counts """
    Rc_reads = read_counts.sum(axis=1)
    Rn_reads = read_counts.sum(axis=0)
   
    ss = time.time()
    dirichlet_prior = opt.optimize_alpha(read_counts, Rc_reads, Rn_reads, n_size)
    print('obtained alpha parameter for read counts', dirichlet_prior, 'in' ,time.time()-ss,'seconds')
    ss = time.time()
    dirichlet_prior_persamples  = dirichlet_prior * Rn_reads / Rn_reads.sum()
    print("dirichlet_prior_persamples", dirichlet_prior_persamples)

    """ generate gamma parameters for Bayes factor in distance calculation """
    
    """ load kmer counts """
    kmer_counts = pd.read_csv(tmp_dir + "kmer_counts", header=None)
    kmer_counts = kmer_counts.to_numpy()
    kmer_counts = kmer_counts.reshape(total_contigs_source, 256) # convert 1D array to a 2D array with {total_contigs, all 4-mers} shape  
    kmer_counts = (kmer_counts / 2)

    # kmer_counts = kmer_counts[longer_contig_inds]

    print("processing kmer frequencies")

    GC_fractions = pd.read_csv(tmp_dir + "GC_fractionof_contigs", header=None)
    GC_fractions = GC_fractions.to_numpy()
    
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
    repeat_index = np.nonzero((kmer_counts / ekmer_counts)>3) # theory suggested z-score but we use ratio for detecting repeat index 
    setind_zero = np.vstack((np.repeat(repeat_index[0],4),\
        np.repeat(repeat_index[1] // 4  * 4, 4) + np.tile([0, 1, 2, 3], np.shape(repeat_index[1]))))
    kmer_counts[setind_zero[0],setind_zero[1]] = 0
    del(ekmer_counts, GC_fractions, setind_zero, repeat_index)


    """ process high kmer counts """
    kmer_counts = kmer_counts[long_contigs]
    # scale_down_kmer = R_max / (R_max + kmer_counts.reshape(-1,64,4).sum(axis=2))
    # kmer_counts = np.multiply(kmer_counts, np.repeat(scale_down_kmer, 4, axis=1))


    trimercountsper_nt = kmer_counts.reshape(-1,64,4).sum(axis=0)
    Rc_kmers = kmer_counts.reshape(-1,64,4).sum(axis=2)

    ss = time.time()
    dirichlet_prior_kmers, dirichlet_prior_perkmers = optimize_prior_fortrimers(kmer_counts, Rc_kmers, trimercountsper_nt)
    print('obtained alpha parameters for kmer counts in', time.time()-ss,'seconds')
    del(trimercountsper_nt)

    contig_length = contig_length[long_contigs]
    contig_names = contig_names[long_contigs]

    # kmer_counts =1
    # Rc_kmers = 1
    # kmer_counts_source = 1
    # dirichlet_prior_kmers = 1
    # dirichlet_prior_perkmers = np.zeros(10)

    # # """ end """

    cluster_parameters = list([read_counts, Rc_reads, contig_length, total_contigs, \
                          dirichlet_prior, dirichlet_prior_persamples, kmer_counts, Rc_kmers, \
                          dirichlet_prior_kmers, dirichlet_prior_perkmers.flatten(), \
                          d0, d1, min_shared_contigs, tmp_dir])   
    # clusters, numclust_incomponents = cluster_by_connecting_centroids(cluster_parameters)
    # clusters, numclust_incomponents = ccm.cluster_by_connecting_centroids(cluster_parameters)
    clusters, numclust_incomponents = dc.cluster_by_connecting_centroids(cluster_parameters)
    with open(tmp_dir + "density_components", 'w+') as file:
        for f in range(len(clusters)):
            for q in clusters[f]:
                file.write(str(q) + " " + str(f) + "\n")
    del(kmer_counts, cluster_parameters)

    # profiler = cProfile.Profile()
    # profiler.enable()

    bins_ = nmf_connected_components(read_counts, contig_length, clusters, numclust_incomponents, dirichlet_prior, dirichlet_prior_persamples)
    
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()
    print(len(np.unique(bins_[1])), "bins obtained in total")
    print(bins_[1])

    np.savetxt(tmp_dir + "bin_assignments", np.stack((contig_names[bins_[0].astype(int)], bins_[1].astype(int))).T, fmt='%s,%d')

    np.savetxt(tmp_dir + "bin_assignments_inds", bins_)

    if args.fasta:
        subprocess.run([os.getcwd() + "/util/get_sequence_bybin " + str(tmp_dir) + " bin_assignments " +str(args.contigs) + " " + str(args.output) + " " + str(args.outdir)], shell=True)

    print(bins_ , time.time()-ss, 'seconds')
    # subprocess.run(["rm " + str(args.outdir) + "/" + "*_seq.fasta"], shell=True)
    # np.savetxt(tmp_dir + "bin_assignments_density_inds", bins_, fmt='%d')

    print(psutil.disk_usage('/'), "disk usage", psutil.cpu_percent(), "cpu percentage")
    print(psutil.swap_memory(), "swap memory", psutil.virtual_memory().percent, "memory percentage")
    print(psutil.virtual_memory(), "virutial memory")
    print(psutil.getloadavg(), "get log of memory load")

    # """ assign short contigs """
    shortcontigs.assign_shortcontigs(tmp_dir, long_contigs_minlength, Rc_reads, contigs, bins_)
    
    print('metagenome binning is completed in', time.time()-s,'seconds')

    gc.collect()
    return 0

if __name__ == "__main__":
    pass
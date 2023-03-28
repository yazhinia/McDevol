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
import nmf_connected_components_modified as nmfm

import bam2counts
from datetime import datetime

def optimize_prior_fortrimers(kmer_counts, Rc_kmers, trimercountsper_nt):

    with Pool() as pool:
        awa_values = pool.starmap(opt.obtain_optimized_alphas, \
                                [(np.take(kmer_counts,[c*4, (c*4)+1, (c*4)+2, (c*4)+3], axis=1),\
                                np.take(Rc_kmers, c, axis=1), trimercountsper_nt[c]) for c in range(64)])
    awa_values = np.array(awa_values)
    alpha_values = awa_values.sum(axis=1)

    return alpha_values, awa_values

# def call_bam2count(*arg):
#     bam2counts.obtain_readcounts(arg[0])

# def calcreadcounts(input_dir, working_dir, minlength):
#     bamfiles = [(f, input_dir, working_dir, minlength) for f in os.listdir(input_dir) if f.endswith('.bam')]
#     with Pool() as pool:
#         pool.map(call_bam2count, bamfiles)

def binning(args):
    input_dir = args.input
    working_dir = args.outdir + '/tmp/'
    s = time.time()
    print(working_dir)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    """ obtain read counts """
    # calcreadcounts(input_dir, working_dir, args.minlength)
    """ obtain kmer counts """
    subprocess.run(["cat " + str(working_dir) + "*_count > " + str(working_dir) + "total_readcount"], shell=True)

    #subprocess.run(["/big/work/metadevol/scripts/kmerfreq " + str(working_dir) + "  " +str(args.contigs)], shell=True)

    """ clustering parameters (default) """
    d0 = 1.0
    d1 = d0
    min_shared_contigs = 100
    
    """ load contig read counts """
    contigs = pd.read_csv(working_dir + 'selected_contigs', header=None, sep=' ').to_numpy()
    contig_names = contigs[:,1]
    contig_length = contigs[:,2].astype(int)

    fractional_counts = pd.read_csv(working_dir + "total_readcount", header=None,sep=' ')
    read_counts = fractional_counts.pivot_table(index = 1, columns = 0, values = 2)
    del(fractional_counts)

    read_counts = read_counts.to_numpy().T
    total_contigs_source = read_counts.shape[0]
    print(total_contigs_source, "shape 0")


    """ remove contigs with no counts """
    # Rc_reads = read_counts.sum(axis=1)
    # read_counts = np.delete(read_counts, np.nonzero(Rc_reads==0)[0], axis=0)

    long_contigs = np.nonzero(contig_length>=2500)[0]
    read_counts = read_counts[long_contigs]
    total_contigs, n_size = np.shape(read_counts)


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
#     ss = time.time()
#     # rand_ind = np.random.randint(total_contigs, size=10000)
#     # gamma_shape, gamma_scale = opt.optimize_gammaparameters(Rc_reads[rand_ind], contig_length[rand_ind], total_contigs)
# #     gamma_shape, gamma_scale = opt.optimize_gammaparameters(Rc_reads, contig_length, total_contigs)
#     print('obtained gamma parameters in', time.time()-ss,'seconds')
    
    """ load kmer counts """
    kmer_counts = pd.read_csv(working_dir + "kmer_counts", header=None)
    kmer_counts = kmer_counts.to_numpy()
    kmer_counts = kmer_counts.reshape(total_contigs_source, 256) # convert 1D array to a 2D array with {total_contigs, all 4-mers} shape  
    kmer_counts = (kmer_counts / 2)

    # kmer_counts = kmer_counts[longer_contig_inds]

    print("processing kmer frequencies")

    GC_fractions = pd.read_csv(working_dir + "GC_fractionof_contigs", header=None)
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

    cluster_parameters = list([read_counts, Rc_reads, total_contigs, \
                          dirichlet_prior, dirichlet_prior_persamples, kmer_counts, Rc_kmers, \
                          dirichlet_prior_kmers, dirichlet_prior_perkmers.flatten(), \
                          d0, d1, min_shared_contigs, working_dir])   
    # clusters, numclust_incomponents = cluster_by_connecting_centroids(cluster_parameters)
    # ccm.cluster_by_connecting_centroids(cluster_parameters)
    clusters, numclust_incomponents = ccm.cluster_by_connecting_centroids(cluster_parameters)
    del(kmer_counts, cluster_parameters)

    # # bins_ = nmf_connected_components(read_counts, contig_length, clusters, numclust_incomponents, dirichlet_prior_persamples, dirichlet_prior)
    bins_ = nmfm.nmf_connected_components(read_counts, contig_length, clusters, numclust_incomponents, dirichlet_prior_persamples, dirichlet_prior)
    print(len(np.unique(bins_[1])), "bins obtained in total")
    print(bins_[1])

    np.savetxt(args.outdir + "/bin_assignments_bothsummed", np.stack((contig_names[bins_[0]], bins_[1])).T, fmt='%s\t%d')
    np.savetxt(args.outdir + "/bin_assignments_bothsummed_inds", bins_, fmt='%d')
    # # # subprocess.run(["/big/work/metadevol/scripts/get_sequence_bybin " + str(working_dir) + "  ../bin_assignments_newalgo" + " " +str(args.contigs) + " " + str(args.output) + " " + str(args.outdir)], shell=True)
    # print('metagenome binning is completed in', time.time()-s,'seconds')


    """ assign short contigs """
    # assign_shortcontigs(working_dir, sel_inds, Rc_reads, contigs, bins_)

    gc.collect()
    return 0

if __name__ == "__main__":
    pass
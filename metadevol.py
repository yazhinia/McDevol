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


# def calcreadcounts(input_dir, working_dir, minlength):
#     print("entering pool process", flush=True)
#     with Pool() as pool:
#         bamfiles = [f for f in os.listdir(input_dir) if f.endswith('.bam')]
#         pool.map(bam2counts.obtain_readcounts,[[str(bam), str(input_dir), str(working_dir), minlength], for bam in bamfiles])
#         # pool.close()
#         # pool.join()


def binning(args):
    working_dir = args.outdir + '/tmp/'
    s = time.time()
    print(working_dir)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    """ obtain read counts """
    # calcreadcounts(input_dir, working_dir, args.minlength)

    """ clustering parameters (default) """
    d0 = 1.0
    d1 = d0
    min_shared_contigs = 5
    
    """ load contig read counts """
    contigs = pd.read_csv(working_dir + 'contigs_labels', header=None, sep=' ').to_numpy()
    contig_names = contigs[:,1]
    contig_length = contigs[:,2].astype(int)


    fractional_counts = pd.read_csv(working_dir + "total_count", header=None,sep=' ')
    read_counts = fractional_counts.pivot_table(index = 1, columns = 0, values = 2)
    # read_counts = pd.read_pickle(working_dir + "X_pickle")
    del(fractional_counts)
    read_counts = read_counts.to_numpy(dtype=np.float32).T
    total_contigs_source = read_counts.shape[0]

    sp80_inds = np.loadtxt(working_dir + '80sp_singlecentroid_inds', dtype=int)
    read_counts = read_counts[sp80_inds]

    total_contigs, n_size = np.shape(read_counts)


    """ process high read counts """
    read_counts_source = read_counts
    R_max = 1e5
    # Rc_reads = np.sum(read_counts, axis=1, keepdims=True)
    # # Rn_reads = read_counts.sum(axis=0)
    # scale_down = R_max/(R_max + Rc_reads)
    # read_counts = np.multiply(read_counts, scale_down)
    Rc_reads = read_counts.sum(axis=1)
    Rn_reads = read_counts.sum(axis=0)

    ss = time.time()
    dirichlet_prior = opt.optimize_alpha(read_counts, Rc_reads, Rn_reads, n_size)
    print('obtained alpha parameter for read counts', dirichlet_prior, 'in' ,time.time()-ss,'seconds')
    ss = time.time()
    dirichlet_prior_persamples  = dirichlet_prior * Rn_reads / Rn_reads.sum()

    """ generate gamma parameters for Bayes factor in distance calculation """
#     ss = time.time()
#     rand_ind = np.random.randint(total_contigs, size=10000)
#     gamma_shape, gamma_scale = opt.optimize_gammaparameters(Rc_reads[rand_ind], contig_length[rand_ind], total_contigs)
# #     gamma_shape, gamma_scale = opt.optimize_gammaparameters(Rc_reads, contig_length, total_contigs)
#     print('obtained gamma parameters in', time.time()-ss,'seconds')
    

    """ load kmer counts """
    kmer_counts = pd.read_csv(working_dir + "kmer_counts", header=None)
    kmer_counts = kmer_counts.to_numpy(dtype=np.float32)
    kmer_counts = kmer_counts.reshape(total_contigs_source, 256) # convert 1D array to a 2D array with {total_contigs, all 4-mers} shape  
    kmer_counts = (kmer_counts / 2)

    # kmer_counts = kmer_counts[longer_contig_inds]

    print("processing kmer frequencies")

    GC_fractions = pd.read_csv(working_dir + "GC_fractionof_contigs", header=None)
    GC_fractions = GC_fractions.to_numpy()
    
    # GC_fractions = GC_fractions[longer_contig_inds]

    # GC_fractions = np.delete(GC_fractions, sel_index, axis=0)
    # print(np.shape(GC_fractions))
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
    kmer_counts_source = kmer_counts
    # scale_down_kmer = R_max / (R_max + kmer_counts.reshape(-1,64,4).sum(axis=2))
    # kmer_counts = np.multiply(kmer_counts, np.repeat(scale_down_kmer, 4, axis=1))

    kmer_counts = kmer_counts[sp80_inds]

    trimercountsper_nt = kmer_counts.reshape(-1,64,4).sum(axis=0)
    Rc_kmers = kmer_counts.reshape(-1,64,4).sum(axis=2)

    ss = time.time()
    dirichlet_prior_kmers, dirichlet_prior_perkmers = optimize_prior_fortrimers(kmer_counts, Rc_kmers, trimercountsper_nt)
    print('obtained alpha parameters for kmer counts in', time.time()-ss,'seconds')
    del(trimercountsper_nt)


    # # """ filter data for selected contigs """

    # sel_index = np.arange(100000) #np.loadtxt(working_dir + 'index_n0cc', dtype=int)
    # read_counts = read_counts[sel_index]
    # Rc_reads = Rc_reads[sel_index]
    # total_contigs = read_counts.shape[0]
    # kmer_counts = kmer_counts[sel_index]
    # Rc_kmers = Rc_kmers[sel_index]
    # contig_length = contig_length[sel_index]
    contig_length = contig_length[sp80_inds]

    # # """ end """

    cluster_parameters = list([read_counts, Rc_reads, total_contigs, \
                          dirichlet_prior, dirichlet_prior_persamples, kmer_counts, Rc_kmers, \
                          dirichlet_prior_kmers, dirichlet_prior_perkmers.flatten(), \
                          d0, d1, min_shared_contigs, working_dir, read_counts_source, kmer_counts_source, R_max])   
    # clusters, numclust_incomponents = cluster_by_connecting_centroids(cluster_parameters)
    # ccm.cluster_by_connecting_centroids(cluster_parameters)
    clusters, numclust_incomponents = ccm.cluster_by_connecting_centroids(cluster_parameters)
    del(kmer_counts, cluster_parameters)

    bins_ = nmf_connected_components(read_counts, contig_length, clusters, numclust_incomponents, dirichlet_prior_persamples, dirichlet_prior)
    print(bins_)
    # np.savetxt(args.outdir + "/bin_assignments_newalgo", np.stack((contig_names[bins_[0]], bins_[1])).T, fmt='%s,%d',delimiter=" ")
    # subprocess.run(["/big/work/metadevol/scripts/get_sequence_bybin " + str(working_dir) + "  ../bin_assignments_newalgo" + " " +str(args.contigs) + " " + str(args.output) + " " + str(args.outdir)], shell=True)
    print('metagenome binning is completed in', time.time()-s,'seconds')
    gc.collect()
    return 0

if __name__ == "__main__":
    pass
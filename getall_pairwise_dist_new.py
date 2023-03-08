import time
import numpy as np
import pandas as pd
import optimize_parameters as opt
from distance_calculations import distance
from multiprocessing.pool import Pool
import metadevol_distance as md
import gc

def optimize_prior_fortrimers(kmer_counts, trimercountsper_nt):

    with Pool() as pool:
        awa_values = pool.starmap(opt.obtain_optimized_alphas, \
                                 [(np.take(kmer_counts,[c*4, (c*4)+1, (c*4)+2, (c*4)+3], axis=1),\
                                  np.take(Rc_kmers, c, axis=1), trimercountsper_nt[c]) for c in range(64)])
    awa_values = np.array(awa_values)
    alpha_values = awa_values.sum(axis=1)
    return alpha_values, awa_values

if __name__ == '__main__':
        
    s = time.time()
    tmp_dir = "/big/work/metadevol/benchmark_dataset1/bamfiles/tmp/"
    # tmp_dir = "/big/work/metadevol/cami2_datasets/marine/bowtie_bamfiles/tmp/"
    # tmp_dir = "/big/work/metadevol/new_simdata/eachsamreads_allcontigs_allalign/tmp/"
    fractional_counts = pd.read_csv(tmp_dir + "total_count", header=None,sep=' ')
    read_counts = fractional_counts.pivot_table(index = 1, columns = 0, values = 2)
    # read_counts = pd.read_pickle(working_dir + "X_pickle")
    del(fractional_counts)
    read_counts = read_counts.to_numpy(dtype=np.float32).T
    contig_length = pd.read_csv(tmp_dir + '../../contigs_umap_hoverdata1', header=None, usecols=[3],sep=' ')
    # contig_length = pd.read_csv(tmp_dir + 'contigs_labels', header=None, usecols=[2],sep=' ')
    contig_length = contig_length.to_numpy().ravel()
    total_contigs, n_size = np.shape(read_counts)
    print(total_contigs)
    
    """ process high read counts """
    Rc_reads = np.sum(read_counts, axis=1, keepdims=True)
    # Rn_reads = read_counts.sum(axis=0)
    R_max = 1e5
    # scale_down = np.minimum(1, R_max/Rc_reads)
    scale_down = R_max/(R_max + Rc_reads)
    read_counts = np.multiply(read_counts, scale_down)
    Rc_reads = read_counts.sum(axis=1)
    Rn_reads = read_counts.sum(axis=0)
    print(np.min(Rc_reads))
    ss = time.time()
    dirichlet_prior = opt.optimize_alpha(read_counts, Rc_reads, Rn_reads, n_size)
    print('obtained alpha parameter in', dirichlet_prior, time.time()-ss,'seconds')
    ss = time.time()
    dirichlet_prior_persamples  = dirichlet_prior * Rn_reads / Rn_reads.sum()

    kmer_counts = pd.read_csv(tmp_dir + "kmer_counts", header=None)
    kmer_counts = kmer_counts.to_numpy(dtype=np.float32)
    kmer_counts = kmer_counts.reshape(total_contigs,256) # convert 1D array to a 2D array with {total_contigs, all 4-mers} shape  
    kmer_counts = (kmer_counts / 2)
    
    
    GC_fractions = pd.read_csv(tmp_dir + "GC_fractionof_contigs", header=None)
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
    repeat_index = np.nonzero((kmer_counts/ekmer_counts)>3)
    setind_zero = np.vstack((np.repeat(repeat_index[0],4),\
        np.repeat(repeat_index[1] // 4  * 4, 4) + np.tile([0, 1, 2, 3], np.shape(repeat_index[1]))))
    kmer_counts[setind_zero[0],setind_zero[1]] = 0
    del(ekmer_counts, GC_fractions, setind_zero, repeat_index)
    
    """ process high read counts """
    scale_down_kmer = R_max / (R_max + kmer_counts.reshape(-1,64,4).sum(axis=2))
    kmer_counts = np.multiply(kmer_counts, np.repeat(scale_down_kmer, 4, axis=1))

    trimercountsper_nt = kmer_counts.reshape(-1,64,4).sum(axis=0) #np.split(kmer_countsT.sum(axis=1), 64)
    Rc_kmers = kmer_counts.reshape(-1,64,4).sum(axis=2) #np.array(trimer_submatrices).sum(axis=1) # kmer_counts.reshape(-1,64,4).sum(axis=2)
    tetramer_count = 256

    ss = time.time()
    dirichlet_prior_kmers, dirichlet_prior_perkmers = optimize_prior_fortrimers(kmer_counts, trimercountsper_nt)
    del(trimercountsper_nt)
    print(dirichlet_prior_kmers, dirichlet_prior_perkmers)


    # """ distance for specific cluster """
    # sel_inds = pd.read_csv(tmp_dir + 'cluster97_allspselindex', header=None).to_numpy(dtype=int).ravel()
    # read_counts_s = read_counts[sel_inds]
    # kmer_counts_s = kmer_counts[sel_inds]
    # Rc_reads_s = Rc_reads[sel_inds]
    # Rc_kmers_s = Rc_kmers[sel_inds]


    # def obtain_distancebyreads(c):
    #     dist = md.compute_readcountdist(c, read_counts, Rc_reads, dirichlet_prior, dirichlet_prior_persamples)
    #     return dist

    # def obtain_distancebykmers(c):
    #     # dist = md.compute_kmercountdist(c, kmer_counts, Rc_kmers, dirichlet_prior_kmers, dirichlet_prior_perkmers)
    #     dist = distance(kmer_counts[c], kmer_counts, Rc_kmers[c], Rc_kmers, dirichlet_prior_perkmers, dirichlet_prior_kmers, 1)
    #     return dist

    def obtain_combineddistance(c):
        # Baysian_factor = calc_Baysian_factor(Rc_reads[c], Rc_reads[c:], contig_length[c], contig_length[c:], gamma_shape, gamma_scale)
        # print(Baysian_factor)
        # distr = distance(read_counts[c], read_counts[c:,], Rc_reads[c], Rc_reads[c:], dirichlet_prior_persamples, dirichlet_prior, 0)#, Baysian_factor)
        # distk = distance(kmer_counts[c], kmer_counts[c:,], Rc_kmers[c], Rc_kmers[c:], dirichlet_prior_perkmers, dirichlet_prior_kmers, 1)
        # dist = distr + distk
        # dist = md.compute_dist(c, read_counts_s, kmer_counts_s, Rc_reads_s, Rc_kmers_s, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers)
        # dist = md.compute_readcountdist(c, read_counts_s, Rc_reads_s, dirichlet_prior, dirichlet_prior_persamples)
        # dist = md.compute_kmercountdist(c, kmer_counts_s, Rc_kmers_s, dirichlet_prior_kmers, dirichlet_prior_perkmers)
        dist = md.compute_dist(c, read_counts, kmer_counts, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers)
        
        return dist

    dist_matrix = []
    
    with Pool() as pool:
        dist_matrix = pool.map(obtain_combineddistance, [c for c in np.arange(total_contigs)])
    np.save(tmp_dir + "X_metadevoldistwithcap.npy",np.array(dist_matrix))
    print("total time took: ", time.time() - s, "seconds")
    del dist_matrix
    gc.collect()

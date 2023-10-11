#!/usr/bin/env python

import os, sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0,'/big/work/mcdevol/scripts/')


parent_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_path)
from util.bayesian_distance import compute_readcountdist as Bayesian_distance
from multiprocessing.pool import Pool
import multiprocessing
from src.optimize_parameters import optimize_alpha
from src.nmf import nmf_cluster
from sklearn.metrics.pairwise import euclidean_distances
import gc
from scipy import stats

from src.bin_assignments import assignment
from scipy.stats import chi2
from util.bayesian_distance import log_gamma_avx2
import src.distance_calculations as dist



global epsilon 
epsilon = 1e-10

def log_gamma(x):
    return log_gamma_avx2(x)

def log_factorial(x):
    return log_gamma(x + 1)

def np_relu(x):
    return np.maximum(0,x)

def entropy_poi(_lambda):
    entropy_cal = 0.5 * np.log(2 * np.pi * 2.718 * _lambda) - (1 / (12 * _lambda)) - (1 / (24 * _lambda **2)) - (19 / (360 * _lambda**3)) 
    return entropy_cal

def initialize_Z(W_t, X):
    W = np.transpose(W_t)
    lmda = 0.1
    Z = np.abs(np.dot(np.linalg.inv(np.matmul(W_t,W) + (lmda * np.eye(W.shape[1]))), np.matmul(W_t, X.T)))
    # Z = np.random.rand(W_t.shape[0], X.shape[0])
    # print(Z, np.min(Z), np.max(Z), Z.shape, 'Z matrix initialized')

    return Z


def maximize_function(w, z, x):

    mean = np.matmul(z.T, w) + epsilon
    negative_log_likelihood = (- mean + np.multiply(x, np.log(mean))).sum()

    return negative_log_likelihood.astype(np.float32)


def calc_aic(w, z, x):

    mean = np.matmul(z.T, w) + epsilon
    log_likelihood = (- mean + np.multiply(x, np.log(mean))).sum()
    AIC = log_likelihood - np.count_nonzero(np_relu(w) > 1e-17) - np.count_nonzero(np_relu(z) > 1e-17)
    
    return AIC

def poisson_mean(x, mean):

    entropy_exp = (x.T * np.log(mean)) - mean - log_factorial(x.T)

    return entropy_exp


def sum_logterm(_lambda):
    log_lamda = np.log(_lambda)
    sum_term = 0.0
    lambda_term = 0.0
    factorial_term = 0.0

    for k in range(1,50):
        lambda_term += log_lamda
        factorial_term += np.log(k)

        sum_term += factorial_term * np.exp(lambda_term - factorial_term)

    return sum_term

def entropy_minival(_lambda):

    return _lambda - (_lambda * np.log(_lambda)) + (np.exp(-_lambda) * sum_logterm(_lambda))

def entropy_maxval(_lambda):
    
    entropy_cal = 0.5 * np.log(2 * np.pi * 2.718 * _lambda) - (1 / (12 * _lambda)) - (1 / (24 * _lambda **2)) - (19 / (360 * _lambda**3)) 
    
    return entropy_cal



""" Multiplicative Updates """

def multiplicative_updates(W, Z, X, n, b):

    log_likelihood = []

    for i in range(n):
        X_by_mean = X / (np.matmul(Z.T, W) + epsilon)
        Z = Z * (np.matmul(W, X_by_mean.T) / W.sum(axis=1)[:,None])
        X_by_mean = X / (np.matmul(Z.T, W) + epsilon)
        W = W * (np.matmul(Z, X_by_mean) / Z.sum(axis=1)[:,None]) 
        log_likelihood.append(maximize_function(W, Z, X))

    np.save('/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/ll_array_'+str(n)+'_'+str(b)+'.npy', log_likelihood)

    AIC_val = calc_aic(W, Z, X)

    return  W, Z, AIC_val, maximize_function(W, Z, X)


def upper_tri_masking(dist_matrix):
    m = dist_matrix.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return dist_matrix[mask]


def obtain_distancebyreads(*args):
    c, read_counts_k, Rc_k, dirichlet_prior, dirichlet_prior_persamples, q_read = args
    dist = Bayesian_distance(c, read_counts_k, Rc_k, dirichlet_prior, dirichlet_prior_persamples, q_read)
    return dist


def get_WZ(U, S, V, read_counts_k, f):

    if np.max(U)<=0 or np.max(V)<=0 or np.max(U)<=0 and np.max(V)<=0:
        print(f'randomly selecting {f} count profiles')
        inds = np.random.choice(read_counts_k.shape[0],f)
        W = read_counts_k[inds]
        Z = initialize_Z(W, read_counts_k)
    else:
        print('else condition')
        if f >= 1:
            W = np.abs(U.dot(np.sqrt(S))).T
            Z = np.abs(np.sqrt(S).dot(V))
        else:
            W = np.abs(U.dot(np.sqrt(S)))
            Z = np.abs(np.sqrt(S).dot(V)).T

    return W, Z


def assign(bval, bin_assigned, optimized, bin_index, k , num_C, contigs_length_k):
    
    if bval == 1:
        bin_assigned.append(np.vstack((k, np.array([bin_index] * num_C))))
        bin_index += 1
            
    else:
        Z_opt = optimized["Z"+str(bval)]
        bin_indices, _ = assignment(Z_opt, contigs_length_k, 0)
        bin_indices += bin_index
        bin_assigned.append(np.vstack((k, bin_indices)))
        bin_index += len(set(bin_indices))
    
    return bin_index, bin_assigned


def nmf_deconvolution(read_counts, clusters, contigs_length, contig_names, dirichlet_prior, dirichlet_prior_persamples):
    counter = 0
    bin_index_ent = 1
    bin_index_aic = 1
    bin_index_ll = 1
    bin_index_md = 1
    bin_assigned_ent = []
    bin_assigned_aic = []
    bin_assigned_ll = []
    bin_assigned_md = []
    ll_values = []
    q = np.exp(-7.0)

    for k in clusters:

        if len(k) > 1:
            num_C = len(k)            
            read_counts_k = read_counts[k]
            read_counts_k = np.delete(read_counts_k, np.nonzero(read_counts_k.sum(axis=0)== 0.0)[0], 1)
            Rc_k = read_counts_k.sum(axis=1)

            contigs_length_k = contigs_length[k]
            optimized = {}            
            AIC_values = []
            ll_values = []
            likelihood_differences = []

            #### b=1 ####
            W = read_counts_k.sum(axis=0)[:,None].T
            Z = initialize_Z(W, read_counts_k)
            W_opt, Z_opt, AIC_val, ll = multiplicative_updates(W, Z, read_counts_k, 2, 1)
            AIC_values.append(AIC_val)
            ll_values.append(ll)
            mean = np.matmul(W_opt.T, Z_opt) + epsilon
            optimized["Z"+str(1)] = Z_opt
            expected_likelihood =  - np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
            observed_likelihood = poisson_mean(read_counts_k, mean)
            likelihood_differences.append(np.mean(observed_likelihood - expected_likelihood))
            #############

            #### b=2 ####
            c1_distance = Bayesian_distance(np.argmax(Rc_k), read_counts_k, Rc_k, dirichlet_prior, dirichlet_prior_persamples, q)
            c1 = np.argmax(c1_distance)
            c2_distance = Bayesian_distance(c1, read_counts_k, Rc_k, dirichlet_prior, dirichlet_prior_persamples, q)
            c2 = np.argmax(c2_distance)
            print(k[[c1,c2]], 'c1 and c2')
            if (c1 == c2):
                print("c1 and c2 are same")
                c2 = np.argsort(c2_distance)[-2]
        
            W =read_counts_k[[c1,c2]]
            Z = initialize_Z(W, read_counts_k)
            W_opt, Z_opt, AIC_val, ll = multiplicative_updates(W, Z, read_counts_k, 10000, 2)
            AIC_values.append(AIC_val)
            ll_values.append(ll)
            mean = np.matmul(W_opt.T, Z_opt) + epsilon
            optimized["Z"+str(2)] = Z_opt
            expected_likelihood = - np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
            observed_likelihood = poisson_mean(read_counts_k, mean)
            likelihood_differences.append(np.mean(observed_likelihood - expected_likelihood))
            likelihood_difference = (observed_likelihood - expected_likelihood).sum(axis=0) / read_counts_k.shape[1]
            lambda_value = 2 * (ll_values[1] - ll_values[0])
            p_value = chi2.sf(np.array(lambda_value), read_counts_k.shape[1] + num_C - 2*2)
            #############
            print(p_value, 'p-value for b= 1')
            inds = [c1,c2]

            if p_value < 0.01:
                
                for f in range(3,10,1):
                    ck_ind = np.argmax(likelihood_difference)
                    if ck_ind in inds:
                        bval = f-1
                        sort_ind = np.argsort(likelihood_difference)
                        sort_ind = np.delete(sort_ind, np.where(sort_ind == ck_ind))
                        for r in inds:
                            sort_ind = np.delete(sort_ind, np.where(sort_ind == r))
                        if len(sort_ind) > 0:
                            ck_ind = sort_ind[-1]
                            inds.append(ck_ind)
                        else:
                            break
                    else:
                        inds.append(ck_ind)

                    W = read_counts_k[inds]
                    Z = initialize_Z(W, read_counts_k)
                    W_opt, Z_opt, AIC_val, ll = multiplicative_updates(W, Z, read_counts_k, 10000, f)
                    AIC_values.append(AIC_val)
                    ll_values.append(ll)
                    mean = np.matmul(W_opt.T, Z_opt) + epsilon
                    optimized["Z"+str(f)] = Z_opt
                    expected_likelihood = - np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
                    observed_likelihood = poisson_mean(read_counts_k, mean)
                    likelihood_differences.append(np.mean(observed_likelihood - expected_likelihood))
                    likelihood_difference = (observed_likelihood - expected_likelihood).sum(axis=0) / read_counts_k.shape[1]
                    lambda_value = 2 * (ll_values[-1] - ll_values[-2])
                    p_value = chi2.sf(np.array(lambda_value), read_counts_k.shape[1] + num_C - 2*f)
                    print(p_value, 'p value for b=', f, flush=True)  
                    
                    if p_value > 0.01:
                        bval = f-1
                        break
                    #############

            else:
                bval = 1

            bval_md = stats.mode([np.argmin(np.abs(likelihood_differences)-0)+1, np.argmax(AIC_values)+1, bval], axis=None, keepdims=True)[0][0]
            print(inds, k[inds], 'inds')
            print(counter, np.argmin(np.abs(likelihood_differences)-0)+1, 'entropy b-value')
            print(counter, np.argmax(AIC_values)+1, 'AIC b-value')
            print(counter, bval, 'log-likelihood b-bvalue')
            print(counter, bval_md, 'mode b-bvalue')
            bin_index_ent, bin_assigned_ent = assign(np.argmin(np.abs(likelihood_differences)-0)+1, bin_assigned_ent, optimized, bin_index_ent, k, num_C, contigs_length_k)
            bin_index_aic, bin_assigned_aic = assign(np.argmax(AIC_values)+1, bin_assigned_aic, optimized, bin_index_aic, k, num_C, contigs_length_k)
            bin_index_ll, bin_assigned_ll = assign(bval, bin_assigned_ll, optimized, bin_index_ll, k, num_C, contigs_length_k)
            bin_index_md, bin_assigned_md = assign(bval_md, bin_assigned_md, optimized, bin_index_md, k, num_C, contigs_length_k)
        
        else:
            bin_assigned_ent.append(np.vstack((k, np.array([bin_index_ent] * len(k)))))
            bin_assigned_aic.append(np.vstack((k, np.array([bin_index_aic] * len(k)))))
            bin_assigned_ll.append(np.vstack((k, np.array([bin_index_ll] * len(k)))))
            bin_assigned_md.append(np.vstack((k, np.array([bin_index_md] * len(k)))))
            bin_index_ent += 1
            bin_index_aic += 1
            bin_index_ll += 1
            bin_index_md += 1

        counter += 1

    
    bin_assigned_ent = np.concatenate(bin_assigned_ent, axis=1).T
    bin_assigned_aic = np.concatenate(bin_assigned_aic, axis=1).T
    bin_assigned_ll = np.concatenate(bin_assigned_ll, axis=1).T
    bin_assigned_md = np.concatenate(bin_assigned_md, axis=1).T

    return bin_assigned_ent, bin_assigned_aic, bin_assigned_ll, bin_assigned_md
        
    #         for f in range(2, range_limit):
                
    #             ##### SVD initialized #######
    #             S_f = np.diag(S[:f])
               
    #             if np.diag(S_f)[-1] != 0.0:
    #                 print('entered f iteration')
    #                 U_f = U[:,:f]
    #                 Vt_f = Vt[:f, :]
                    
    #                 W, Z = get_WZ(U_f, S_f, Vt_f, read_counts_k, f)

    #                 W_opt, Z_opt, AIC_val, ll = multiplicative_updates(W, Z, read_counts_k, 500, f)
    #                 ##############################


    #                 # ###### sklearn approach ######
    #                 # NMF_model = NMF(n_components=f, init='nndsvdar', solver='mu', beta_loss='kullback-leibler', max_iter=10000)
    #                 # W_opt = NMF_model.fit_transform(read_counts_k.T).T
    #                 # Z_opt = NMF_model.components_
    #                 # ##############################

    #                 AIC_values.append(calc_aic(W_opt, Z_opt, read_counts_k))
    #                 ll_values.append(maximize_function(W_opt, Z_opt, read_counts_k))
                    
    #                 mean = np.matmul(W_opt.T, Z_opt) + epsilon
    #                 optimized["Z"+str(f-1)] = Z_opt
    #                 entropy_exp = np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
    #                 entropy_obs = poisson_mean(read_counts_k, mean)
    #                 entropy_difference = np.mean(entropy_obs + entropy_exp)
    #                 entropy_differences.append(entropy_difference)
    #             else:
    #                 break
            
    #         bval = np.argmin(np.abs(entropy_differences)-0) # np.argmax(AIC_values) # np.argmin(np.abs(entropy_differences)-0)
    #         counter_bvalnmf.append([counter, bval])
    #         print(counter, bval, entropy_differences, 'entropy b-value')
    #         print(counter, np.argmax(AIC_values), 'AIC b-value')
    #         # np.save(tmp_dir + '/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/Z_opt.npy', optimized["Z"+str(bval)])
            

    #         lambda_values = [2 * (ll_values[i+1] - ll_values[i]) for i in range(len(ll_values)-1)]


    #         p_valuelist = []

    #         for B, lambda_value in enumerate(lambda_values, start=2):
    #             p_value = chi2.sf(np.array(lambda_value), 10 + num_C - 2*B)
    #             p_valuelist.append(p_value)
    #             print(B, 'B value', p_value, 'p value')
    #             if p_value > 0.05:
    #                 B -= 2
    #                 break
    #             else:
    #                 B += 1
    #         if B > len(ll_values)-1:
    #             B = 0

    #         bval = B
    #         print(counter, B, 'log-likelihood b-bvalue')

    #         if bval == 0:
    #             bin_assigned.append(np.vstack((k, np.array([bin_index] * num_C))))
    #             bin_index += 1
            
    #         else:
    #             Z_opt = optimized["Z"+str(bval)]
    #             bin_indices, _ = assignment(Z_opt, contigs_length_k, 0)
    #             bin_indices += bin_index
    #             bin_assigned.append(np.vstack((k, bin_indices)))
    #             bin_index += len(set(bin_indices))    

    #     # else:
    #     #     bin_assigned.append(np.vstack((k, np.array([bin_index] * len(k)))))
    #     #     bin_index += 1

    # #     counter += 1

    
    # # bin_assigned = np.concatenate(bin_assigned, axis=1).T
    # # return bin_assigned

    # with open(tmp_dir+'/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/Mcdevol_bins_i'+str(flag), 'w+') as file:
    #     for q in bin_assigned:
    #         file.write(str(contig_names[q[0]]) + "," + str(q[1]) + "\n")

    # bins_list = np.split(bin_assigned[:,0], np.unique(bin_assigned[:, 1], return_index=True)[1][1:])
    # # print(bins_list, 'bin list')
    # bins = []
    
    # for i, j in enumerate(bins_list):
    #     for k in j:
    #         bins.append([k, i])
    
    # bins=np.array(bins).astype(int)
    
    # return counter_bvalnmf, bins_list, bins

if __name__ == "__main__":
    s = time.time()
    variable = 'autoencoder/reads_andcorrkmers_twoaug'
    tmp_dir = "/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/"
    read_counts = np.load('/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/mcdevol_readcounts.npz', allow_pickle=True)['arr_0']
    contigs_length = np.load('/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/minimap_process/mcdevol_run/contigs_2klength.npz', allow_pickle=True)['arr_0']
    
    Rc_reads = read_counts.sum(axis=1)
    Rn_reads = read_counts.sum(axis=0)
    ss = time.time()
    dirichlet_prior = optimize_alpha(read_counts, Rc_reads, Rn_reads)
    print('obtained alpha parameter for read counts', dirichlet_prior, 'in' ,time.time()-ss,'seconds')
    dirichlet_prior_persamples  = dirichlet_prior * Rn_reads / Rn_reads.sum()

    contig_names = np.load(tmp_dir + 'contigs_2knames.npz', allow_pickle=True)['arr_0']
    clusters_ip = pd.read_csv(tmp_dir+'/autoencoder/reads_andcorrkmers_twoaug/clusters_cosine.tsv', header=None, sep='\t')
    mapped = pd.DataFrame(contig_names).merge(clusters_ip, left_on=0, right_on=1)[[1,'0_y']]
    mapped['0_y'] = [i for i in mapped['0_y']] # [int(i.split('_')[-1]) for i in mapped['0_y']]
    mapped = mapped.sort_values('0_y')
    clusters = mapped.groupby('0_y', sort=False).apply(lambda x: np.array(x.index)).to_numpy()


    bin_assigned_ent, bin_assigned_aic, bin_assigned_ll, bin_assigned_md = nmf_deconvolution(read_counts, clusters, contigs_length, contig_names, dirichlet_prior, dirichlet_prior_persamples)

    
    # print('run time: ', time.time()-s, 'seconds first round')
    # _, bins_list2, bins_2 = nmf_deconvolution(read_counts, bins_list1, contigs_length, contig_names, False)
    # print(bins_1, bins_2)

    # np.save(tmp_dir + '/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/bins_1.npy', bins_1)
    # np.save(tmp_dir + '/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/bins_2.npy', bins_2)
    
    with open(tmp_dir+'/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/Mcdevol_bins_baysd_ent', 'w+') as file:
        for q in bin_assigned_ent:
            file.write(str(contig_names[q[0]]) + "," + str(q[1]) + "\n")
    with open(tmp_dir+'/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/Mcdevol_bins_baysd_aic', 'w+') as file:
        for q in bin_assigned_aic:
            file.write(str(contig_names[q[0]]) + "," + str(q[1]) + "\n")
    with open(tmp_dir+'/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/Mcdevol_bins_baysd_ll', 'w+') as file:
        for q in bin_assigned_ll:
            file.write(str(contig_names[q[0]]) + "," + str(q[1]) + "\n")
    with open(tmp_dir+'/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/Mcdevol_bins_baysd_md', 'w+') as file:
        for q in bin_assigned_md:
            file.write(str(contig_names[q[0]]) + "," + str(q[1]) + "\n")

    # with open(tmp_dir+'/autoencoder/reads_andcorrkmers_twoaug/nmf_investigate/Mcdevol_bins_i2', 'w+') as file:
    #     for q in bins_2:
    #         file.write(str(contig_names[q[0]]) + "," + str(q[1]) + "\n")

    print('run time: ', time.time()-s, 'seconds')    



"""
nmf = nimfa.Nmf(read_counts_k, max_iter=10000, rank=1, seeding='nndsvd' ,update='divergence', objective='div')
# nmf_fit = nmf()
# W_opt = nmf_fit.basis().T
# Z_opt = nmf_fit.coef().T
# AIC_values.append(calc_aic(W_opt, Z_opt, read_counts_k))
# ll_values.append(maximize_function(W_opt, Z_opt, read_counts_k))
# mean = np.matmul(Z_opt, W_opt).T + epsilon
# print(W_opt.shape, Rc_k.shape, Z_opt.shape, mean.shape, 'W and Z')
# optimized["Z"+str(1)] = W_opt.T
# # entropy_exp = np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
# # entropy_obs = poisson_mean(read_counts_k, mean)
# # entropy_differences.append(np.mean(entropy_obs + entropy_exp))

# nmf = nimfa.Nmf(read_counts_k, max_iter=10000, rank=2, seeding='nndsvd' ,update='divergence', objective='div')
# nmf_fit = nmf()
# W_opt = nmf_fit.basis().T
# Z_opt = nmf_fit.coef().T
# AIC_values.append(calc_aic(W_opt, Z_opt, read_counts_k))
# ll_values.append(maximize_function(W_opt, Z_opt, read_counts_k))
# mean = np.matmul(Z_opt, W_opt).T + epsilon
# optimized["Z"+str(2)] = W_opt.T
# # entropy_exp = np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
# # entropy_obs = poisson_mean(read_counts_k, mean)
# # entropy_differences.append(np.mean(entropy_obs + entropy_exp))

# nmf = nimfa.Nmf(read_counts_k, max_iter=10000, rank=3, seeding='nndsvd' ,update='divergence', objective='div')
# nmf_fit = nmf()
# W_opt = nmf_fit.basis().T
# Z_opt = nmf_fit.coef().T
# AIC_values.append(calc_aic(W_opt, Z_opt, read_counts_k))
# ll_values.append(maximize_function(W_opt, Z_opt, read_counts_k))
# mean = np.matmul(Z_opt, W_opt).T + epsilon
# optimized["Z"+str(3)] = W_opt.T
# # entropy_exp = np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
# # entropy_obs = poisson_mean(read_counts_k, mean)
# # entropy_differences.append(np.mean(entropy_obs + entropy_exp))

# nmf = nimfa.Nmf(read_counts_k, max_iter=10000, rank=4, seeding='nndsvd' ,update='divergence', objective='div')
# nmf_fit = nmf()
# W_opt = nmf_fit.basis().T
# Z_opt = nmf_fit.coef().T
# AIC_values.append(calc_aic(W_opt, Z_opt, read_counts_k))
# ll_values.append(maximize_function(W_opt, Z_opt, read_counts_k))
# mean = np.matmul(Z_opt, W_opt).T + epsilon
# optimized["Z"+str(4)] = W_opt.T
# # entropy_exp = np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
# # entropy_obs = poisson_mean(read_counts_k, mean)
# # entropy_differences.append(np.mean(entropy_obs + entropy_exp))

# nmf = nimfa.Nmf(read_counts_k, max_iter=10000, rank=5, seeding='nndsvd' ,update='divergence', objective='div')
# nmf_fit = nmf()
# W_opt = nmf_fit.basis().T
# Z_opt = nmf_fit.coef().T
# AIC_values.append(calc_aic(W_opt, Z_opt, read_counts_k))
# ll_values.append(maximize_function(W_opt, Z_opt, read_counts_k))
# mean = np.matmul(Z_opt, W_opt).T + epsilon
# optimized["Z"+str(5)] = W_opt.T
# # entropy_exp = np.where(mean<=5, entropy_minival(mean), entropy_maxval(mean))
# # entropy_obs = poisson_mean(read_counts_k, mean)
# # entropy_differences.append(np.mean(entropy_obs + entropy_exp))

# lambda_values = [2 * (ll_values[i+1] - ll_values[i]) for i in range(len(ll_values)-1)]

# p_valuelist = []

# for B, lambda_value in enumerate(lambda_values, start=2):
#     p_value = chi2.sf(np.array(lambda_value), 10 + num_C - 2*B)
#     p_valuelist.append(p_value)
#     print(B, 'B value', p_value, 'p value')
#     if p_value > 0.05:
#         B -= 2
#         break
#     else:
#         B += 1
# if B > len(ll_values)-1:
#     B = 0

# bval = B
"""
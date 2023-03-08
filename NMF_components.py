import time
import numpy as np
import pandas as pd
from distance_calculations import distance
from new_clustering_algorithm import cluster_by_connecting_centroids


def np_relu(x):
    
    return np.maximum(0,x)


def initialize_Z(W, dat):
    
    W_t = np.transpose(W)
    lmda = 0.1
    inverse_term = np.linalg.inv(np.eye(W.shape[0]) + (lmda ** -1) * np.matmul(W,W_t))
    woodbury = (lmda ** -1) * np.eye(W.shape[1]) - np.matmul((lmda ** -2) * W_t , np.matmul(inverse_term, W))
    
    Z = np_relu(np.matmul(woodbury, np.matmul(W_t,dat)))
    
    return Z


def maximize_function(w, z, x):
    
    mean = np.matmul(w, z)
    mean = 1e-10 + np_relu(mean)
    negative_log_likelihood = np.sum(- mean + np.multiply(x, np.log(mean)))

    return negative_log_likelihood


def calc_aic(w, z, x):

    mean = np.matmul(w, z)
    mean = 1e-10 + np_relu(mean)
    log_likelihood = np.sum(- mean + np.multiply(x, np.log(mean)))
    # print('differences', np.count_nonzero(np_relu(w) > 1e-05) - np.count_nonzero(np_relu(z) > 1e-05))
    AIC = log_likelihood - np.count_nonzero(np_relu(w) > 1e-05) - np.count_nonzero(np_relu(z) > 1e-05)
    
    return AIC


""" Multiplicative Updates """

def multiplicative_updates(W, Z, X, n, method):
    
    lw = 0.0
    lz = 1e-01
    beta = 0.5
    convergence_criteria = 0.0001
    epsilon_reg = 1e-05
    loss_values = []
    
    global AIC
    
    if method == 0:
        
        for i in range(n):

            mean = np.matmul(W, Z)
            mean = 1e-10 + np_relu(mean)
            X_by_mean = X / mean

            W = np.multiply(W, np_relu(np.matmul(X_by_mean, np.transpose(Z))))
            W = W / np.sum(W, axis = 0)
            Z = np.multiply(Z, np_relu(np.matmul(np.transpose(W), X_by_mean)))

            loss_values.append(maximize_function(W, Z, X))
            
            if len(loss_values) >= 10 :
                
                if ((loss_values[i] - loss_values[i-10]) / loss_values[i]) < convergence_criteria:
                    AIC = calc_aic(W, Z, X)
                    # print("Function is converged")
                    break
        return  W, Z, loss_values, AIC
    
    
    if method == 1:
        
        for i in range(n):

            mean = np.matmul(W, Z)
            mean = 1e-10 + np_relu(mean)
            X_by_mean = X / mean

            Z_reg =  (lz * beta) / np.power(np.abs(Z+epsilon_reg), 1-beta)
            Z = np.multiply(Z, np_relu(np.matmul(np.transpose(W), X_by_mean) - Z_reg))
            
            loss_values.append(maximize_function(W, Z, X))

            if len(loss_values) >= 10 :
                if ((loss_values[i] - loss_values[i-10]) / loss_values[i]) < convergence_criteria:
                    AIC = calc_aic(W, Z, X)
                    # print("Function is converged")
                    break
                
        return  Z, loss_values, AIC
    
    
    if method == 2:
        
        for i in range(n):

            mean = np.matmul(W, Z)
            mean = 1e-10 + np_relu(mean)
            X_by_mean = X / mean

            W_reg =  (lw * beta) / np.power(np.abs(W+epsilon_reg), 1-beta)
            Z_reg =  (lz * beta) / np.power(np.abs(Z+epsilon_reg), 1-beta)

            W = np.multiply(W, np_relu(np.matmul(X_by_mean, np.transpose(Z)) - W_reg))
            W = W / np.sum(W, axis = 0) 
            Z = np.multiply(Z, np_relu(np.matmul(np.transpose(W), X_by_mean) - Z_reg))

            loss_values.append(maximize_function(W, Z, X))

            # if len(loss_values) >= 10 :
            #     if ((loss_values[i] - loss_values[i-10]) / loss_values[i]) < convergence_criteria:
            #         AIC = calc_aic(W, Z, X)
            #         # print("Function is converged")
            #         break
                
        return  W, Z, loss_values, AIC

    
""" End multiplicative updates """
                               
                                
if __name__ == "__main__":
    
    s = time.time()
    print("clustering initiated"+'\n')
    # tmp_dir = "/big/work/metadevol/benchmark_dataset1/"
    tmp_dir = "/big/work/metadevol/scripts/bamtools_api/build/"
    dat = pd.read_pickle(tmp_dir + 'X_pickle')
    contig_names = dat.columns
    d0 = 1.0
    min_shared_contigs = 10
    s_start = time.time()
    clusters, numclust_incomponents, an, alpha, N = cluster_by_connecting_centroids(dat, d0, min_shared_contigs)
    print("distance_calculation:", time.time() - s_start)
    print("overall time taken for new clustering is: ", time.time()-s)
    
    # label = pd.read_csv(tmp_dir + 'contigs_ids_distindex', sep =' ', header = None)
    label = pd.read_csv(tmp_dir + 'contigs_refid_genomeid_withcrossmappeddata', sep =' ', header = None)
    lc = label[3]
    
    dat = dat.to_numpy()
   
    num_iterations = 100
    
    W_full = []
    
    epsilon = 1e-10


    for k in np.arange(len(numclust_incomponents)):

        if numclust_incomponents[k] > 1 :
        
            dat_s = dat[:, clusters[k]]

            Rc_s = dat_s.sum(axis = 0)
            
            AIC_score = []
            
            optimized = {}
            
            LL_values = {}
            
            for _ in range(3):
                
                trial_bval = _
                k_contigs = []

                if trial_bval == 0:
                    k_contigs = dat_s.sum(axis=1) + epsilon
                    k_contigs = k_contigs.reshape(N,1)

                else:
                    if trial_bval + 1 <= numclust_incomponents[k]: 

                        ind = []

                        for i in range(trial_bval+1):

                            if i == 0:

                                c0_ind = np.argmax(dat_s.sum(axis=0))
                                c0 = dat_s[:, [c0_ind]]
                                dist = distance(c0, dat_s, Rc_s[c0_ind], Rc_s, N, an, alpha)
                                c_ind = np.argsort(dist)[-1]
                                c = dat_s[:, [c_ind]]
                                k_contigs = c
                                ind.append(c_ind)

                            else:

                                dist_sum = []

                                for f in range(np.shape(k_contigs)[1]):
                                    dist_sum.append(distance(k_contigs[:,[f]], dat_s, Rc_s[ind[f]], Rc_s, N, an, alpha))
                                
                                f_ind = np.argmax(np.array(dist_sum).sum(axis=0))
                                ind.append(f_ind)
                                k_contigs = np.hstack((k_contigs, dat_s[:, [f_ind]])) 

                W = np.array(k_contigs) 
                
                W_norm = W / W.sum(axis = 0)
            
                if not np.shape(W)[0] == 0:
                    
                    Z = initialize_Z(W_norm, dat_s)

                    print(maximize_function(W_norm , Z, dat_s), "pre NMF negative loglikelihood")

                    W_opt, Z_opt, LL, AIC = multiplicative_updates(W_norm, Z, dat_s, num_iterations, 0)
                    optimized["W"+str(trial_bval)] = W_opt
                    optimized["Z"+str(trial_bval)] = Z_opt
                    AIC_score.append(AIC)
                    
                    LL_values[str(trial_bval)] = LL
                
            bval = np.argmax(AIC_score)
            W_full.append(optimized["W"+str(bval)])


        else:

            dat_s = dat[:, clusters[k]].sum(axis=1) + epsilon
            dat_s = dat_s.reshape(N,1) 
            dat_s = dat_s / dat_s.sum() # not resulting in 1.0 (eg., 0.9999999999999998, 1.0000000000000002)
            W_full.append(dat_s)

    W_full = np.concatenate(W_full, axis = 1)

    Z = initialize_Z(W_full, dat)
    Z_full, LL, AIC_score = multiplicative_updates(W_full, Z, dat, num_iterations, 1)
       
    # W_fullopt, Z_fullopt, LL, AIC_final = multiplicative_updates(W_full, Z_full, dat, num_iterations, 2)
    
    
    
    # """ Assignment """
    # Z_assign = Z_full
    # Rc_c  = np.sum(Z_assign, axis=0)
    # pb_c  = Z_assign / Rc_c
    # cov_b = np.sum(Z_assign, axis=1) / np.sum((np.array(lc) * Z_assign) / Rc_c, axis=1)
    # pb_min = 0.8 * (cov_b.reshape(len(cov_b),1) * np.sum(np.square(pb_c), axis=0) \
    #                 / np.sum(cov_b.reshape(len(cov_b),1) * pb_c, axis=0))
    # pb_min[pb_min > 0.5] = 0.5
    # contig_assign0 = np.argmax(pb_c/pb_min, axis=0)
    
    # print(len(set(contig_assign0)), "number of bins")
    # print(len(set(np.argmax(Z_assign, axis=0))), "just max")


    # """ New assignment - more than one bins for contigs is possible"""
    Z_assign = Z_full
    Rc_c  = np.sum(Z_assign, axis=0)
    pb_c  = np.power(Z_assign/Rc_c,4) 
    Z_l = Z_assign / np.array(lc)

    cov_b = (np.multiply(pb_c,Z_l).sum(axis=1)) / pb_c.sum(axis=1)

    pi_bc1 = Z_assign / np.row_stack(cov_b)
    pi_bc2 = pi_bc1.sum(axis=0)

    pi_bc = pi_bc1 / pi_bc2

    np.argwhere(pi_bc>0.2)
    # np.savetxt(tmp_dir + "indices_pibc_bins",np.argwhere(pi_bc>0.2), fmt="%d")
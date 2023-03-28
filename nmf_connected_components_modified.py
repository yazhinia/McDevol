import numpy as np
import distance_calculations as dist
import bin_assignments as assign
from multiprocessing.pool import Pool
from os import getpid
import logging

def np_relu(x):
    
    return np.maximum(0,x)


def initialize_Z(W, X):
    
    W_t = np.transpose(W)
    lmda = 0.1
    inverse_term = np.linalg.inv(np.eye(W.shape[1]) + (lmda ** -1) * np.matmul(W_t,W))
    woodbury = (lmda ** -1) * np.eye(W.shape[0]) - np.matmul((lmda ** -2) * W , np.matmul(inverse_term, W_t))
    Z = np_relu(np.matmul(np.matmul(woodbury, W), X.T))
    
    return Z


def maximize_function(w, z, x):
    
    mean = np.matmul(z.T, w)
    mean = 1e-10 + np_relu(mean)
    negative_log_likelihood = (- mean + np.multiply(x, np.log(mean))).sum()

    return negative_log_likelihood


def calc_aic(w, z, x):

    mean = np.matmul(z.T, w)
    mean = 1e-10 + np_relu(mean)
    log_likelihood = (- mean + np.multiply(x, np.log(mean))).sum()
    AIC = log_likelihood - np.count_nonzero(np_relu(w) > 1e-05) - np.count_nonzero(np_relu(z) > 1e-05)
    
    return AIC


""" Multiplicative Updates """

def multiplicative_updates(W, Z, X, n, method):
    
    lw = 0.0
    lz = 0.8
    beta = 0.5
    convergence_criteria = np.exp(-15)
    epsilon_reg = 1e-05
    loss_values = []

    if method == 0:
        
        for i in range(n):
            
            mean = np.matmul(Z.T, W)
            mean = 1e-10 + np_relu(mean)
            X_by_mean = X / mean
            
            W = np.multiply(W, np_relu(np.matmul(Z, X_by_mean)))
            W = W / np.array([W.sum(axis = 1)]).T
            Z = np.multiply(Z, np_relu(np.matmul(W, X_by_mean.T)))

            loss_values.append(maximize_function(W, Z, X))
            
            if len(loss_values) >= 10 :
                
                if ((loss_values[i] - loss_values[i-10]) / loss_values[i]) < convergence_criteria:
                    AIC = calc_aic(W, Z, X)
                    # print("Function is converged")
                    break
                else:
                    raise RuntimeError(f'function not converged') 
        print(AIC, "within method mul update")
        return  Z, AIC
    
    
    if method == 1:
        
        for i in range(n):

            mean = np.matmul(Z.T, W)
            mean = 1e-10 + np_relu(mean)
            X_by_mean = X / mean

            Z_reg =  (lz * beta) / np.abs(Z+epsilon_reg) ** (1-beta)
            Z = np.multiply(Z, np_relu(np.matmul(W, X_by_mean.T) - Z_reg))
            
            loss_values.append(maximize_function(W, Z, X))

            if len(loss_values) >= 10 :
                if ((loss_values[i] - loss_values[i-10]) / loss_values[i]) < convergence_criteria:
                    AIC = calc_aic(W, Z, X)
#                     print("Function is converged")
                    break
                else:
                    raise RuntimeError(f'function not converged')    
                
        return  Z, AIC    

    if method == 2:
            
        for i in range(n):

            mean = np.matmul(Z.T, W)
            mean = 1e-10 + np_relu(mean)
            X_by_mean = X / mean

            W_reg =  (lw * beta) / np.abs(W+epsilon_reg) ** (1-beta)
            Z_reg =  (lz * beta) / np.abs(Z+epsilon_reg) ** (1-beta)

            W = np.multiply(W, np_relu(np.matmul(Z, X_by_mean) - W_reg))
            W = W / np.array([W.sum(axis = 1) ]).T
            Z = np.multiply(Z, np_relu(np.matmul(W, X_by_mean.T) - Z_reg))

            loss_values.append(maximize_function(W, Z, X))

            if len(loss_values) >= 10 :
                if ((loss_values[i] - loss_values[i-10]) / loss_values[i]) < convergence_criteria:
                    AIC = calc_aic(W, Z, X)
                    # print("Function is converged")
                    break
                else:
                    raise RuntimeError(f'function not converged')
                          
        return  Z, AIC
    
    raise ValueError(f'method must be 0, 1, 2, got {method}')


def obtain_best_noofbins(argv):
    
    dat_s = argv[0]
    epsilon = argv[1]
    Rc_s = argv[2]
    an = argv[3]
    alpha = argv[4]
    num_iterations = argv[5]
    AIC_score = argv[6]
    optimized = argv[7]
    LL_values = argv[8]
    trial_bval = argv[9]
    
    if trial_bval == 0:

        k_contigs = np.array([dat_s.sum(axis=0) + epsilon])

    else:

        k_contigs = []
        ind = []

        c0_ind = np.argmax(dat_s.sum(axis=1))
        c0 = dat_s[c0_ind]

        for i in range(trial_bval+1):

            if i == 0:
                distance = dist.distance(c0, dat_s, Rc_s[c0_ind], Rc_s, alpha, an, 0)
                c_ind = np.argsort(distance)[-1]
                c = dat_s[c_ind]
                k_contigs = [c]
                ind.append(c_ind)

            else:

                dist_sum = []

                for f in range(len(np.array([k_contigs]))):
                    dist_sum.append(dist.distance(k_contigs[f], dat_s, Rc_s[ind[f]], Rc_s, alpha, an, 0))

                f_inds = np.argsort(np.array(dist_sum).sum(axis=0))
                check_ind = np.append(ind, c0_ind)

                if len(np.setdiff1d(f_inds, check_ind)) > 0:
                    f_ind = np.setdiff1d(f_inds, check_ind)[-1]
                    ind.append(f_ind)
                    k_contigs.append(dat_s[f_ind])   
    
    W = np.array(k_contigs)
    W_norm = W / np.array([W.sum(axis = 1)]).T

    if not np.shape(W)[0] == 0:

        Z = initialize_Z(W_norm, dat_s)

        Z_opt, _AIC = multiplicative_updates(W_norm, Z, dat_s, num_iterations, 0)
        optimized["Z"+str(trial_bval)] = Z_opt
        # AIC_score.append(_AIC)

    return _AIC, optimized


def nmf_connected_components(read_counts, contig_length, clusters, numclust_incomponents, an, alpha, small_nmfassign_flag=1, mode = 0):
    
    num_iterations = 5000
    
    W_full = []
    epsilon = 1e-10
    bin_index = 1
    bin_assigned = []
    k_contigs = []
    b_value_record = []

    for k in np.arange(len(numclust_incomponents)):
               
        if numclust_incomponents[k] > 1 :

            dat_s = read_counts[clusters[k],:]
            Rc_s = dat_s.sum(axis = 1)

            AIC_score = []

            optimized = {}

            LL_values = {}

            _params = [(dat_s, epsilon, Rc_s, an, alpha, num_iterations, \
                        AIC_score, optimized, LL_values, trial_bval) \
                       for trial_bval in range(numclust_incomponents[k])]

            with Pool() as pool:
                results = pool.map(obtain_best_noofbins, _params)
            print(k, numclust_incomponents[k], len(np.array([x[0] for x in results])), np.array([x[0] for x in results]))
            bval = np.argmax(np.array([x[0] for x in results]))
            Z_opt = [x[1]['Z'+str(bval)] for x in results if 'Z'+str(bval) in x[1]][0]

            b_value_record.append(list([k,bval]))

            if small_nmfassign_flag == 1:

                if bval == 0:

                    bin_assigned.append(np.vstack((clusters[k], np.array([bin_index] * len(clusters[k])))))

                    bin_index += 1

                else:

                    print(k, bval, numclust_incomponents[k], bin_index)

                    bin_indices, bin_mindices = assign.assignment(Z_opt, contig_length[clusters[k]], 0)
                    bin_indices += bin_index

                    bin_mindices_ = bin_mindices[0] + bin_index

                    if mode == 1:
                        bin_assigned.append(np.vstack((clusters[k][bin_mindices[1]], bin_mindices_)))
                    else:
                        bin_assigned.append(np.vstack((clusters[k], bin_indices)))

                    bin_index += len(set(bin_indices))

        else:

            b_value_record.append(list([k,0]))
            print(k, "only one cluster")
            if len(clusters[k]) == 0:
                print("warning")
            if contig_length[clusters[k]].sum() >= 10000:
                dat_s = np.array([read_counts[clusters[k]].sum(axis=0) + epsilon])
                dat_s = dat_s / dat_s.sum()
                bin_assigned.append(np.vstack((clusters[k], np.array([bin_index] * len(clusters[k])))))
                bin_index += 1

    bin_assigned = np.concatenate(bin_assigned, axis=1)
    # np.savetxt("/big/work/metadevol/cami2_datasets/marine/bowtie_bamfiles/tmp/try/length_2000/bvalue_list_summed",b_value_record, fmt="%d\t%d")
    return bin_assigned
    

if __name__ == "__main__" :
    
    pass
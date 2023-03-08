import numpy as np
from distance_calculations import distance
import bin_assignments as assign

def np_relu(x):
    
    return np.maximum(0,x)


def initialize_Z(W, X):
    
    W_t = np.transpose(W)
    lmda = 0.1
    inverse_term = np.linalg.inv(np.eye(W.shape[1]) + (lmda ** -1) * np.matmul(W_t,W))
    woodbury = (lmda ** -1) * np.eye(W.shape[0]) - np.matmul((lmda ** -2) * W , np.matmul(inverse_term, W_t))
    # print(np.shape(woodbury), np.shape(np.matmul(woodbury, W)), np.shape(X.T), flush=True)
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
    lz = 0.1
    beta = 0.5
    convergence_criteria = 0.0001
    epsilon_reg = 1e-05
    loss_values = []
    
    global AIC
    
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
#                     print("Function is converged")
                    break
        return  W, Z, loss_values, AIC
    
    
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
                
        return  Z, loss_values, AIC    

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
                    
            return  Z, loss_values, AIC


def nmf_connected_components(read_counts, contig_length, clusters, numclust_incomponents, an, alpha, small_nmfassign_flag=1, mode = 0):
    
    num_iterations = 100
    
    W_full = []
    epsilon = 1e-10
    bin_index = 1
    bin_assigned = []
    for k in np.arange(len(numclust_incomponents)):

        if numclust_incomponents[k] > 1 :
            
            dat_s = read_counts[clusters[k],:]
            Rc_s = dat_s.sum(axis = 1)
            AIC_score = []
            
            optimized = {}
            
            LL_values = {}
            
            
            for trial_bval in range(3):
                
                if trial_bval == 0:
                    
                    k_contigs = np.array([dat_s.sum(axis=0) + epsilon])

                else:
                    
                    if trial_bval + 1 <= numclust_incomponents[k]: 

                        ind = []
                        k_contigs = []
                        
                        for i in range(trial_bval+1):

                            if i == 0:
                                c0_ind = np.argmax(dat_s.sum(axis=1))
                                c0 = dat_s[c0_ind]
                                dist = distance(c0, dat_s, Rc_s[c0_ind], Rc_s, an, alpha, 0)
                                c_ind = np.argsort(dist)[-1]
                                c = dat_s[c_ind]
                                k_contigs = [c]
                                ind.append(c_ind)

                            else:

                                dist_sum = []

                                for f in range(len(np.array([k_contigs]))):
                                    dist_sum.append(distance(k_contigs[f], dat_s, Rc_s[ind[f]], Rc_s, an, alpha, 0))
                                f_inds = np.argsort(np.array(dist_sum).sum(axis=0))
                                check_ind = np.append(ind, c0_ind)
                                f_ind = np.setdiff1d(f_inds, check_ind)[-1]
                                ind.append(f_ind)
                                k_contigs.append(dat_s[f_ind])    

                W = np.array(k_contigs)
                W_norm = W / np.array([W.sum(axis = 1)]).T
                
                if not np.shape(W)[0] == 0:
                    
                    Z = initialize_Z(W_norm, dat_s)

                    W_opt, Z_opt, LL, AIC = multiplicative_updates(W_norm, Z, dat_s, num_iterations, 0)
                    optimized["W"+str(trial_bval)] = W_opt
                    optimized["Z"+str(trial_bval)] = Z_opt
                    AIC_score.append(AIC)
                    LL_values[str(trial_bval)] = LL

            # obtain best k using maximum AIC value    
            bval = np.argmax(AIC_score)
            W_full.append(optimized["W"+str(bval)])
            # print(k, bval, numclust_incomponents[k])
            if small_nmfassign_flag == 1:
                if bval == 0:
                    bin_assigned.append(np.vstack((clusters[k], [bin_index] * len(clusters[k]))))
                else:
                    Z_opt = optimized["Z"+str(bval)]
                    bin_indices, bin_mindices = assign.assignment(Z_opt, contig_length[clusters[k]],0)
                    bin_indices += bin_index
                    bin_mindices_ = bin_mindices[0] + bin_index
                    if mode == 1:
                        bin_assigned.append(np.vstack((clusters[k][bin_mindices[1]], bin_mindices_)))
                    else:
                        bin_assigned.append(np.vstack((clusters[k], bin_indices)))
                bin_index += bval+1
        else:
            # print(k, "only one cluster")
            if len(clusters[k]) == 0:
                print("warning")
            if contig_length[clusters[k]].sum() >= 100000:
                # print(k, "satisfies if condition")
                dat_s = np.array([read_counts[clusters[k]].sum(axis=0) + epsilon])
                dat_s = dat_s / dat_s.sum()
                W_full.append(dat_s)
                bin_assigned.append(np.vstack((clusters[k], [bin_index] * len(clusters[k]))))
                bin_index += 1

    bin_assigned = np.concatenate(bin_assigned, axis=1)
    # W_full = np.concatenate(W_full, axis = 0)
    # Z = initialize_Z(W_full, read_counts)
    # Z_full, LL, AIC_score = multiplicative_updates(W_full, Z, read_counts, num_iterations, 1)
    # print(np.shape(W_full), np.shape(Z_full))
    # Z_final, LL, AIC_score = multiplicative_updates(W_full, Z_full, read_counts, num_iterations, 2)
    
    return bin_assigned

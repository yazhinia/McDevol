#!/usr/bin/env python3

import numpy as np
from scipy import special

# import simdefy
# simdefy.init()

# import simdefynewavx2
# simdefynewavx2.init()

# def log_gamma(x):
#     if not x.flags['C_CONTIGUOUS']:
#         print(x.flags,'is not continuous')
#     return simdefy.log_gamma_avx2(x)


# def log_gamma(x):
#     if not x.flags['C_CONTIGUOUS']:
#         print(x.flags,'is not continuous')
#     return simdefynewavx2.log_gamma_avx2(x)


def log_gamma(x):
    return special.gammaln(x)



def xn_alpha(x, x_prime, an, flag):
    kk1 = log_gamma(x + an)  + log_gamma(x_prime + an)
    kk2 = log_gamma(an)  + log_gamma(x + x_prime + an)
    if flag == 1:
        xn_alpha = (kk1-kk2).reshape(-1,64,4).sum(axis=2)
    else:
        if (kk1-kk2).ndim == 1:
            xn_alpha = (kk1 - kk2).sum()
        else:
            xn_alpha = (kk1 - kk2).sum(axis=1)
    
    # print("xn_alpha term", sum(log_gamma(x + an)  + log_gamma(x_prime + an) - log_gamma(x + x_prime + an)), sum(log_gamma(an)))

    return xn_alpha

def R_alpha(R, R_prime, alpha):
    R_alpha1 = log_gamma(alpha) + log_gamma(R + R_prime + alpha)
    R_alpha2 = log_gamma(R + alpha) + log_gamma(R_prime + alpha)

    # print("R term", R_alpha1 - R_alpha2, log_gamma(R + alpha) + log_gamma(R_prime + alpha) - log_gamma(R + R_prime + alpha) - log_gamma(alpha) )

    return R_alpha1 - R_alpha2


def calc_Bayes_factor(*argv):
    R = argv[0]
    R_prime = argv[1]
    lc = argv[2]
    lc_prime = argv[3]
    gamma_shape = argv[4]
    gamma_scale = argv[5]

    R_term1 = log_gamma(R + gamma_shape) + log_gamma(R_prime + gamma_shape)
    R_term2 = log_gamma(gamma_shape) + log_gamma(R + R_prime + gamma_shape)
    lc_term1 = np.multiply((R + R_prime + gamma_shape ), np.log(lc + lc_prime + gamma_scale))
    lc_term2 = np.multiply((R + gamma_shape), np.log(lc + gamma_scale)) + np.multiply((R_prime + gamma_shape), np.log(lc_prime + gamma_scale))
    return (R_term1 - R_term2) + (lc_term1 - lc_term2)


def distance(*argv):
    xc = argv[0]
    xc_p = argv[1]
    Rc = argv[2]
    Rc_p = argv[3]
    an = argv[4]
    alpha = argv[5]
    flag = argv[6]
    q = np.exp(-8)
    if flag == 1:
        q = 0.6 #np.exp(-2)
        # rep_w_matrix = argv[7]
    pq = np.log((1.0 - q)/ q)

    if flag == 1:
        R_and_Xnterm = R_alpha(Rc, Rc_p, alpha) + xn_alpha(xc, xc_p, an.flatten(), flag)
        # R_and_Xnterm = (1 - rep_w_matrix) * R_and_Xnterm
        dist = np.logaddexp(np.log(1), pq + R_and_Xnterm.sum(axis=1))
    else:
        R_and_Xnterm = R_alpha(Rc, Rc_p, alpha) + xn_alpha(xc, xc_p, an, flag)
        dist = np.logaddexp(np.log(1), pq + R_and_Xnterm)
        # dist = np.logaddexp(np.log(1), pq + R_and_Xnterm + argv[7]) # without Bayes factor not used
    return dist


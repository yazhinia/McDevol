#!/usr/bin/env python

from scipy import special
import numpy as np
import time

def xn_alpha(x, x_prime, an):
    kk1 = special.gammaln(x + an)  + special.gammaln(x_prime + an)
    kk2 = special.gammaln(an)  + special.gammaln(x + x_prime + an)
    return (kk1 - kk2).sum(axis=0)

def R_alpha(R, R_prime, alpha):
    R_alpha1 = special.gammaln(alpha) + special.gammaln(R + R_prime + alpha)
    R_alpha2 = special.gammaln(R + alpha) + special.gammaln(R_prime + alpha)
    return R_alpha1 - R_alpha2


def distance(*argv):
    xc = argv[0]
    xc_p = argv[1]
    Rc = argv[2]
    Rc_p = argv[3]
    samples = argv[4]
    an = argv[5]
    alpha = argv[6]

    q = np.exp(-4)
    pq = np.log((1 - q)/ q)

    if np.shape(xc_p)[0] != samples:
        dist = np.logaddexp(np.log(1), pq + R_alpha(Rc, Rc_p, alpha) + xn_alpha(xc.reshape(samples,1), xc_p.reshape(samples,1), an.reshape(samples,1)))
    else:
        dist = np.logaddexp(np.log(1), pq + R_alpha(Rc, Rc_p, alpha) + xn_alpha(xc.reshape(samples,1), xc_p, an.reshape(samples,1)))
    return dist

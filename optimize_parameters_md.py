#!/usr/bin/env python

__doc__ = "optimize alpha value using maximum likelihood estimation"

import numpy as np
from scipy import special
from scipy.optimize import minimize, minimize_scalar
import metadevol_distance as md

import simdefy
simdefy.init()

# log_gamma = special.gammaln

def log_gamma(x):
#     return simdefy.log_gamma_avx2(x)
    return md.log_gamma_avx2(x)

def factorial(x):
    return md.log_gamma_avx2(x + 1)

def train_alpha(a, *argv):
    Rc_factorial = factorial(argv[1])
    xn_factorial = factorial(argv[0]).sum(axis=1)
    first_term = Rc_factorial - xn_factorial
    an = np.exp(a) * argv[2] / argv[2].sum()
    second_term = log_gamma(an.sum()) - log_gamma(an).sum()
    third_term = log_gamma(argv[0] + an).sum(axis=1) - log_gamma(argv[1] + an.sum())
    maximize_term = first_term + second_term + third_term
    return -maximize_term.sum()

def optimize_alpha(*argv):
    """ optimize alpha for dirichlet prior distribution using maximum likelihood estimation """
    fun = lambda a: train_alpha(a, *argv)
    optimized_alpha = minimize_scalar(fun, method = 'brent')
    return np.exp(optimized_alpha.x)
    
def train_alphabeta(param, *argv):
    a, b = param
    a = np.exp(a)
    b = np.exp(b)
    Rc_reads = argv[0]
    contig_length = argv[1]
    first_term = log_gamma(Rc_reads + a)
    second_term = log_gamma(a)
    third_term = a * np.log(b/contig_length)
    fourth_term = (Rc_reads + a) * np.log(1 + (b/contig_length))
    maximize_term = first_term - second_term + third_term - fourth_term
    return -maximize_term.sum()

def optimize_gammaparameters(Rc_reads, contig_length, total_contigs):
    """ optimize alpha and beta values for gamma distribution using maximum likelihood estimation """
    a = 0.8
    Rc_dividedby_length = (Rc_reads / contig_length).sum() / total_contigs
    b = a * 1 / Rc_dividedby_length
    optimized_gammaparameters = minimize(train_alphabeta, [a, b], args = (Rc_reads, contig_length), method = 'nelder-mead')
    return np.exp(optimized_gammaparameters.x)
    
def obtain_optimized_alphas(*argv):
    """ optimize alpha values for dirichlet distributions of 64 trimers using maximum likelihood estimation """
    counts = np.array(argv[2])
    aw = optimize_alpha(argv[0], argv[1], counts, 4)
    aw = aw * counts / counts.sum()
    return aw

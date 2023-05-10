#!/usr/bin/env python

import os, sys
parent_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_path)

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from util.bayesian_distance import log_gamma_avx2

def log_gamma(x):
    return log_gamma_avx2(x)

def factorial(x):
    return log_gamma(x + 1)

def train_alpha(a, counts, Rc, Rn):
    Rc_factorial = factorial(Rc)
    xn_factorial = factorial(counts).sum(axis=1)
    first_term = Rc_factorial - xn_factorial

    an = np.exp(a) * Rn / Rn.sum()

    second_term = log_gamma(an.sum()) - log_gamma(an).sum()
    third_term = log_gamma(counts + an).sum(axis=1) - log_gamma(Rc + an.sum())
    maximize_term = first_term + second_term + third_term
    return -maximize_term.sum()

def optimize_alpha(counts, Rc, Rn):
    
    __doc__ = "optimize alpha for dirichlet prior distribution using maximum likelihood estimation"

    fun = lambda a: train_alpha(a, counts, Rc, Rn)
    optimized_alpha = minimize_scalar(fun, method = 'brent')
    return np.exp(optimized_alpha.x)
    
def train_alphabeta(param, Rc_reads, contig_length):
    a, b = param
    a = np.exp(a)
    b = np.exp(b)
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
    
def obtain_optimized_alphas(kmer_counts, Rc_kmers, trimercountsper_nt):
    """ optimize alpha values for dirichlet distributions of 64 trimers using maximum likelihood estimation """
    counts = np.array(trimercountsper_nt)
    aw = optimize_alpha(kmer_counts, Rc_kmers, counts)
    aw = aw * counts / counts.sum()
    return aw

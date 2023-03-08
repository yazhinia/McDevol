import numpy as np

def assignment(Z_bc, contig_length, mode):
    
    if mode == 0:
        Rc  = np.sum(Z_bc, axis=0)
        weights = Z_bc ** 5 / Rc ** 4
        cov_b1 = np.sum(weights  * (Z_bc / contig_length) , axis = 1)
        cov_b2 = weights.sum(axis=1)
        cov_b = cov_b1 / cov_b2
        pi_bc = (Z_bc / cov_b[:,None]) / (Z_bc / cov_b[:,None]).sum(axis = 0)
        bins = np.argmax(pi_bc, axis=0)
        bins_m = np.nonzero(pi_bc >= np.max(pi_bc, axis=0) * 0.6)
    
    if mode == 1:
        Rc  = np.sum(Z_bc, axis=0)
        weights = Z_bc / Rc
        selected_inds = np.nonzero(weights>0.95)[1]
        cov_b1 = (Z_bc[:,selected_inds] / contig_length[selected_inds]).sum(axis=1)
        cov_b2 = Z_bc[:,selected_inds].sum(axis=1)
        cov_b = cov_b1 / cov_b2
        pi_bc = (Z_bc / cov_b[:,None]) / (Z_bc / cov_b[:,None]).sum(axis = 0)
        bins = np.argmax(pi_bc, axis=0)

        # cov_b1m = (weights[:,selected_inds] * (Z_bc[:,selected_inds] / contig_length[selected_inds])).sum(axis=1)
        # cov_b2m = (weights[:,selected_inds] * Z_bc[:,selected_inds]).sum(axis=1)
        # cov_bm = cov_b1m / cov_b2m
        # pi_bcm = (Z_bc / cov_bm[:,None]) / (Z_bc / cov_bm[:,None]).sum(axis = 0)
        bins_m = np.nonzero(pi_bc >= np.max(pi_bc, axis=0) * 0.6)

    return bins, bins_m
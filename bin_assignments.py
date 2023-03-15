import numpy as np

def get_binindex(bins):
    
    sequential_index = dict(np.vstack([np.unique(bins),np.arange(len(np.unique(bins)))]).T)
    bins = [sequential_index[x] for x in bins]

    return np.array(bins)



def assignment(Z_bc, contig_length, mode):
    
    Rc  = np.sum(Z_bc, axis=0)
    
    if mode == 0:
        weights = Z_bc ** 5 / Rc ** 4
        cov_b1 = np.sum(weights  * (Z_bc / contig_length) , axis = 1)
        cov_b2 = weights.sum(axis=1)
        cov_b = cov_b1 / cov_b2
        pi_bc = (Z_bc / cov_b[:,None]) / (Z_bc / cov_b[:,None]).sum(axis = 0)
        bins = np.argmax(pi_bc, axis=0)
        bins = get_binindex(bins)
        bins_m = np.nonzero(pi_bc >= np.max(pi_bc, axis=0) * 0.6)
    
    else:

        weights = Z_bc / Rc
        selected_inds = np.nonzero(weights>0.95)[1]
        print(len(selected_inds), "length of selected_inds")
        
        if selected_inds.size != 0:

            cov_b1 = (Z_bc[:,selected_inds] / contig_length[selected_inds]).sum(axis=1)
            cov_b2 = Z_bc[:,selected_inds].sum(axis=1)
        
        else:

            cov_b1 = (Z_bc / contig_length).sum(axis=1)
            cov_b2 = Z_bc.sum(axis=1)
        print(cov_b2)
        cov_b = cov_b1 / cov_b2
        pi_bc = (Z_bc / cov_b[:,None]) / (Z_bc / cov_b[:,None]).sum(axis = 0)
        bins = np.argmax(pi_bc, axis=0)


        bins_m = np.nonzero(pi_bc >= np.max(pi_bc, axis=0) * 0.6)

    return bins, bins_m
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
        # pi_bc[pi_bc < 0.5] = 0.0
        # poorprob_inds = np.nonzero(pi_bc.sum(axis=0)==0)[0]
        # print(len(poorprob_inds), np.shape(pi_bc), flush=True)
        # pi_bc = np.delete(pi_bc, poorprob_inds, axis=1)
        print(np.max(pi_bc), np.max(pi_bc, axis=0), flush=True)
        np.save('/big/work/metadevol/cami2_datasets/marine/pooled_assembly/pi_bc',pi_bc)
        bins = np.argmax(pi_bc, axis=0)
        bins = get_binindex(bins)
        bins_m = np.nonzero(pi_bc >= np.max(pi_bc, axis=0) * 0.6)
    
    else:

        weights = Z_bc / Rc
        selected_inds = np.nonzero(weights>0.95)[1]
        
        if selected_inds.size > 0:

            print(selected_inds, Z_bc[:,selected_inds])
            cov_b1 = (Z_bc[:,selected_inds] ** 2 / contig_length[selected_inds]).sum(axis=1)
            cov_b2 = Z_bc[:,selected_inds].sum(axis=1)
            print(cov_b2)
        
        else:

            cov_b1 = (Z_bc ** 2 / contig_length).sum(axis=1)
            cov_b2 = Z_bc.sum(axis=1)

        cov_b = cov_b1 / cov_b2
        pi_bc = (Z_bc / cov_b[:,None]) / (Z_bc / cov_b[:,None]).sum(axis = 0)
        
        bins = np.argmax(pi_bc, axis=0)
        bins = get_binindex(bins)

        bins_m = np.nonzero(pi_bc >= np.max(pi_bc, axis=0) * 0.6)

        # weights = 1 / (1 + np.exp(-10 * ((Z_bc / Rc ) - 0.9)))

        # cov_b1 = (weights * (Z_bc ** 2 / contig_length)).sum(axis=1)
        # cov_b2 = (weights * Z_bc).sum(axis=1)

        # cov_b = cov_b1 / cov_b2
        # pi_bc = (Z_bc / cov_b[:,None]) / (Z_bc / cov_b[:,None]).sum(axis = 0)

        # # np.max(pi_bc, axis=0)
        # # pi_bc[pi_bc < 0.7] = 0.0
        # # poorprob_inds = np.nonzero(pi_bc.sum(axis=0)==0)[0]
        # bins = np.argmax(pi_bc, axis=0)
        # bins = get_binindex(bins)

        # bins_m = np.nonzero(pi_bc >= np.max(pi_bc, axis=0) * 0.6)


    return bins, bins_m
import pandas as pd
import numpy as np
import bin_assignments as assign
import nmf_connected_components as nmf

def assign_shortcontigs(tmp_dir, long_contigs_minlength, Rc_reads, contigs, bins_):

    contig_names = contigs[:,1]
    contig_length = contigs[:,2].astype(int)
    fractional_counts = pd.read_csv(tmp_dir + "total_readcount", header=None,sep=' ', engine="pyarrow")
    read_counts = fractional_counts.pivot_table(index = 1, columns = 0, values = 2)
    del(fractional_counts)

    print("here")
    read_counts = read_counts.to_numpy().T

    sel_inds = np.nonzero((contig_length>=1000) & (contig_length<long_contigs_minlength))[0]
    # sel_inds = np.nonzero(contig_length<long_contigs_minlength)[0]

    used_inds = np.nonzero(contig_length>=long_contigs_minlength)[0]

    read_counts_sel =  read_counts[used_inds]

    read_counts_n = read_counts[sel_inds]
    
    del(read_counts)


    bins_ = bins_.T

    print(bins_)
    Rc_reads_bins = Rc_reads[bins_[:,0].astype(int)]

    if np.min(Rc_reads_bins) != 0.0:
        
        bins_withRc = np.column_stack((bins_[:,0], bins_[:,1], Rc_reads_bins.T)).astype(int)
        bins_withRc = pd.DataFrame(bins_withRc)
        bins_withRc.columns = ['ind','bin','Rc']
        bin_selectedinds = bins_withRc.loc[bins_withRc.groupby('bin').Rc.idxmax()]['ind']

        W_bins = read_counts_sel[bin_selectedinds] / read_counts_sel[bin_selectedinds].sum(axis=1, keepdims=True)

        
        Z = nmf.initialize_Z(W_bins, read_counts_n)

        contig_length_n = contig_length[sel_inds]
        contig_names_n = contig_names[sel_inds]
        contig_names = contig_names[used_inds]

        split_count = 3
        Z_parts = np.array_split(Z, split_count, axis=1)
        read_counts_npart = np.array_split(read_counts_n, split_count, axis=0)
        Z_optimized = []
        AIC_values = []
        
        print(np.shape(read_counts_n), len(contig_length_n))

        for f in range(split_count):
            Z_opt, AIC = nmf.multiplicative_updates(W_bins, Z_parts[f], read_counts_npart[f], 1000, 0, 1)
            Z_optimized.append(Z_opt)
            AIC_values.append(AIC)

        Z_optimized = np.concatenate(Z_optimized, axis=1)
        print(np.shape(Z_optimized), "Z_optimized")
        bin_assign, bin_pi, bin_assignmulti = assign.assignment(Z_optimized, contig_length_n, 0)
        short_addedbins = np.stack((contig_names_n, bin_assign)).T
        np.savetxt(tmp_dir + 'short_addedbins', short_addedbins, fmt='%s,%d')
        # total_bins = np.vstack([initial_bins, short_addedbins])

        # return total_bins

    else:
        
        raise RuntimeError("some contigs may have zero total count. Filter them before processing")
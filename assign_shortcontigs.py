import pandas as pd
import numpy as np
import bin_assignments as assign
import nmf_connected_components_modified as nmf

def assign_shortcontigs(working_dir, sel_inds, Rc_reads, contigs, bins_):
    contig_names = contigs[:,1]
    contig_length = contigs[:,2].astype(int)
    fractional_counts = pd.read_csv(working_dir + "total_readcount", header=None,sep=' ', engine="pyarrow")
    read_counts = fractional_counts.pivot_table(index = 1, columns = 0, values = 2)
    del(fractional_counts)

    read_counts = read_counts.to_numpy().T
    read_counts_sel =  read_counts[sel_inds]

    read_counts_n = np.delete(read_counts, sel_inds, axis=0)
    del(read_counts)

    bins_ = bins_.T

    Rc_reads_bins = Rc_reads[bins_[:,0]]

    if np.min(Rc_reads_bins) != 0.0:
        
        bins_withRc = np.column_stack((bins_, Rc_reads_bins.T)).astype(int)
        bins_withRc = pd.DataFrame(bins_withRc)
        bins_withRc.columns = ['ind','bin','Rc']
        bin_selectedinds = bins_withRc.loc[bins_withRc.groupby('bin').Rc.idxmax()]['ind']

        W_bins = read_counts_sel[bin_selectedinds] / read_counts_sel[bin_selectedinds].sum(axis=1, keepdims=True)
        
        Z = nmf.initialize_Z(W_bins, read_counts_n)

        contig_length_n =np.delete(contig_length, sel_inds)
        contig_names_n = np.delete(contig_names, sel_inds)
        contig_names = contig_names[sel_inds]

        split_count = 3
        Z_parts = np.array_split(Z, split_count, axis=1)
        read_counts_npart = np.array_split(read_counts_n, 10, axis=0)
        Z_optimized = []
        AIC_values = []
        
        for f in range(split_count):
            Z_opt, AIC = nmf.multiplicative_updates(W_bins, Z_parts[f], read_counts_npart[f], 1000, 0)
            Z_optimized.append(Z_opt)
            AIC_values.append(AIC)

        Z_optimized = np.concatenate(Z_optimized, axis=1)
        bin_assign = assign.assignment(Z_optimized, contig_length_n, contig_names_n, 0)
        initial_bins = np.stack((contig_names[bins_[0]], bins_[1])).T
        short_addedbins = np.stack((contig_names_n[bin_assign[0]], bin_assign[1])).T
        total_bins = np.vstack([initial_bins, short_addedbins])

        return total_bins

    else:
        
        raise RuntimeError("some contigs may have zero total count. Filter them before processing")
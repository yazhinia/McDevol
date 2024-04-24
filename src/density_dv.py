#!/usr/bin/env python

import os
import sys
import heapq
import numpy as np
import pandas as pd
import hnswlib
import time
import matplotlib.pyplot as plt
from connected_components import find_connected_components, dijkstra_max_min_density, dijkstra_max_min_density1
from collections import Counter
from multiprocessing import Pool, current_process
from functools import partial
import gc
import plotly.express as px
import pdb

# local memory check
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


KUPPA = 0.4 # rough idea of how many density peaks given C total contigs
TOTAL_CONTIGS = 0
LATENT_DIMENSION = 0
ETA = 0 # 1/latent_dim weight of contigs length
DISTANCE_CUTOFF = 1
NN_SIZE_CUTOFF = 500000
NCPUS = (os.cpu_count() if os.cpu_count() is not None else 8)

def add_selfindex(data_indices, labels, distances):
    """ add self index where missed by hnswlib """

    # in case query index does not found as first nearest neighbor,
    # insert manually and remove last nearest neighbor
    add_selfinds = np.nonzero(labels.astype(int)[:,0] != data_indices)[0]
    labels[add_selfinds] = np.insert(labels[add_selfinds], 0, add_selfinds, axis=1)[:,:-1]
    distances[add_selfinds] = np.insert(distances[add_selfinds], 0, 0.0, axis=1)[:,:-1]

    return labels, distances

def get_neighbors(latent, length):
    """ Get neighbors and distance cutoff"""
    s = time.time()
    data_indices = np.arange(TOTAL_CONTIGS)
    k_nn = 4 # int(TOTAL_CONTIGS ** KUPPA)

    p = hnswlib.Index(space = 'l2', dim = LATENT_DIMENSION)
    # M - max. number of out-edges ef_construction
    # ef_construction - the size of the dynamic list for the nearest neighbors
    # and controls index_time/accuracy. Important during indexing time
    p.init_index(max_elements = TOTAL_CONTIGS, ef_construction = 200, M = 16)
    p.set_ef(k_nn) # number should be greater than k_nn
    p.set_num_threads(NCPUS)
    p.add_items(latent)
    print(k_nn, 'k_nn neighbors')
    retry = True
    while retry:
        labels, distances = p.knn_query(latent, k=k_nn)
        cumsum_length = np.cumsum(length[labels], axis=1)
        contigs_with_enough_neighbors = np.nonzero(cumsum_length[:,-1] > NN_SIZE_CUTOFF)[0]
        retry = len(contigs_with_enough_neighbors) / TOTAL_CONTIGS < 0.5
        if retry:
            k_nn = int(k_nn * 1.5) # adjustable
            p.set_ef(k_nn) # adjustable
    del cumsum_length

    labels, distances = add_selfindex(data_indices, labels, distances)
    dist_cutoff_indices = np.argmax(
            np.cumsum(length[labels], axis=1) > NN_SIZE_CUTOFF, axis=1)
    indices_tofind_dist = np.nonzero(dist_cutoff_indices)[0]
    # for long contigs dist_cutoff_indices would be zero. Hence, add manually
    ind_check_longcontig = np.nonzero(dist_cutoff_indices==0)[0]
    indices_to_add = np.nonzero(length[ind_check_longcontig]> NN_SIZE_CUTOFF)[0]
    # consider only contigs that have neighbors with cumsum length of 200kb
    indices_tofind_dist = np.concatenate((indices_tofind_dist, \
        ind_check_longcontig[indices_to_add]))

    # Find distance cutoff
    global DISTANCE_CUTOFF
    DISTANCE_CUTOFF = np.median(
        distances[indices_tofind_dist,dist_cutoff_indices[indices_tofind_dist]])
    # del indices_to_add, indices_tofind_dist, ind_check_longcontig, dist_cutoff_indices
    print(DISTANCE_CUTOFF, np.sqrt(DISTANCE_CUTOFF), 'distance cutoff')

    distance_percontig = distances[data_indices,dist_cutoff_indices]
    distance_percontig[distance_percontig == 0.0] = DISTANCE_CUTOFF
    nnbydist_indices = [[] for _ in data_indices]
    for i, inds in enumerate(dist_cutoff_indices):
        nn_inds = labels[i,:inds+1]
        nnbydist_indices[i] = nn_inds
        # long contig of at least nnsize, or
        # contig has neighbor list with nnsize beyond distance cutoff
        # restrict neighbor list
        # print(otuids[i], otuids[nn_inds])
        # if len(nn_inds) < 4 or distance_percontig[i] > DISTANCE_CUTOFF:
        #     idx = np.nonzero(distances[i] <= DISTANCE_CUTOFF)[0]
        #     nnbydist_indices[i] = labels[i][idx]
        #     distance_percontig[i] = DISTANCE_CUTOFF
        if length[i] < 8000:
            # contigs with many neighbors but short and hence noisely include many neighbors
            nnbydist_indices[i] = labels[i][:10]
            distance_percontig[i] = distances[i,10]
        # else:
        #     nnbydist_indices[i] = nn_inds

    # distances_denominator = [distances[c][1] \
    #         if distances[c][1] < DISTANCE_CUTOFF \
    #         else DISTANCE_CUTOFF for c in data_indices]

    densities = np.array([length[c] ** ETA / distance_percontig[c] \
                   for c in data_indices])
    # Densities with neighbors data, sum_nn(l_nni/d_nni)
    # NOTE: still nearby shorter contigs have higher density than longest contigs in the neighbor set
    # densities = np.array([length[c]** ETA / (0.5 * distances_denominator[c]) +\
    #                 sum(length[nnbydist_indices[c][1:]] ** ETA / distances[c, 1 :len(nnbydist_indices[c])]) \
    #                 for c in data_indices])
    
    # with open('nnlist_knnstart4', 'w', encoding='utf8') as file1:
    #     for i, nn in enumerate(nnbydist_indices):
    #         file1.write(str(i) + ' ' + str(otuids[i]) + ' ' \
    #         + str(contig_length[i]) + ' '+ str(densities[i]) \
    #         + ' ' + str(len(nn))+'\n')
    #         otuids_list = Counter(otuids[nn])
    #         for otu, count in otuids_list.items():
    #             file1.write(str(otu)+'\t'+str(count)+'\n')
    #         file1.write('\n')
    # file1.close()
    # print('writing completed')

    graph = {c : list(zip(nnbydist_indices[c][1:], densities[nnbydist_indices[c][1:]]))
            if len(nnbydist_indices[c][1:]) > 0 \
            else [] for c in data_indices}

    gc.collect()

    return nnbydist_indices, distance_percontig, labels, distances, densities, graph

def get_density_peaks(nnbydist_indices, labels, length, densities):
    """ compute density using k-nearest neighbors """
    s = time.time()

    nearest = np.full(TOTAL_CONTIGS,-1) # np.arange(TOTAL_CONTIGS)
    density_peaks_flag = np.full(TOTAL_CONTIGS,-1)

    indices = np.argsort(length)[::-1]

    for c in indices:
        neighbor_inds = nnbydist_indices[c]
        maxdensity_point = np.argmax(densities[neighbor_inds])
        if maxdensity_point == 0:
            density_peaks_flag[c] = c
            nearest[c] = c
        else:
            # print(otuids[c], otuids[neighbor_inds[maxdensity_point]])
            # print(length[c], length[neighbor_inds[maxdensity_point]])
            nearest[c] = neighbor_inds[maxdensity_point]

    density_peaks_inds = np.nonzero(density_peaks_flag>=0)[0]

    def trace_nearest(nearest):
        """ trace nearest of nearest to assign points to density peaks """
        nearest_prev = np.full(nearest.size, -1, dtype=int)
        while (nearest != nearest_prev).any():
            nearest_prev = nearest
            nearest = nearest[nearest[:]]
        return nearest

    nearest = trace_nearest(nearest)


    # Sometimes tracing back through nearest of nearest
    # end up linking one density peak to another
    # Retain only density peaks that are in the traced-back nearest
    counts = Counter(nearest)
    peak_nonn  = np.array(list(set(density_peaks_inds) - set(counts.keys())))
    flag = any(True if n == nearest[n] or nearest[n] < 0 \
        else False for n in peak_nonn)
    if flag:
        raise RuntimeWarning("peak with no neigbor \
        has not been linked with another peak")
    density_peaks_inds = np.setdiff1d(density_peaks_inds, peak_nonn)

    if len(set(nearest)) != len(density_peaks_inds):
        raise RuntimeWarning("peaks in nearest and \
        total density peaks do not match")

    get_inds = np.argsort(length[density_peaks_inds])[::-1] # [:peak_count]
    density_peaks = density_peaks_inds[get_inds]

    print(time.time() - s, len(density_peaks), \
    'total density peaks', 'seconds for graph and densities calculation')
    gc.collect()

    # plt.figure(figsize=(16,10))
    # plt.plot(np.sort(densities[density_peaks]), marker='o', linewidth=0.0)

    # # plt.plot(np.sort(densities[density_peaks_selected]), marker='o', linewidth=0.0)
    # plt.xlabel('points')
    # plt.ylabel('density')
    # plt.savefig('densities_ofpeaks_subset.png', dpi=600, format='png', bbox_inches = 'tight')

    primary_clusters = []
    for nn in density_peaks:
        inds = np.nonzero(nearest == nn)[0]
        primary_clusters.append(inds)

    with open('nearest_list', 'w', encoding='utf8') as file1:
        for i, nn in enumerate(density_peaks):
            inds = np.nonzero(nearest == nn)[0]
            for idx in inds:
                file1.write(str(i) + ' ' + str(nn) + ' ' + str(otuids[idx]) + ' ' \
                + str(length[idx]) + ' '+ str(densities[idx]) +'\n')
            file1.write('\n')
    file1.close()
    # np.save('density_peaks_subset.npy', density_peaks)
    # np.save('densities_subset.npy', densities)
    print(len(density_peaks), len(set(nearest)), 'density peaks and nearest count')
    return densities, density_peaks, nearest, primary_clusters

def process_peak(i, latent, graph, densities, density_peaks, \
    contig_length, primary_clusters, nnbydist_indices):
    """ process each peak to find density valley """
    peak = density_peaks[i]
    if density_peaks[i] != density_peaks[i:][0]:
        raise RuntimeError('subset peaks doesn\'t include self peak')
    distances_topeaks = np.sum((latent[density_peaks[i:]] - latent[peak])**2, axis=1)
    # filtered_indices = np.nonzero(distances_topeaks <= DISTANCE_CUTOFF*100)[0]
    # query_peaks_indices = filtered_indices[np.argsort(distances_topeaks[filtered_indices])] + i
    # if len(query_peaks_indices) > 1:
    #     print(peak, density_peaks[query_peaks_indices], 'peak and query density peaks')
    query_peaks_indices = np.arange(len(latent[density_peaks[i:]])) + i


    nodes = np.concatenate([primary_clusters[j] for j in query_peaks_indices]).ravel()
    subgraph = {node: graph[node] for node in nodes}

    for node in subgraph:
        node_edges = subgraph[node]
        filtered_edges = [(edge, weight) for edge, weight in node_edges if edge in nodes]
        if filtered_edges:
            subgraph[node] = filtered_edges
        else:
            subgraph[node] = []

    if peak not in subgraph.keys():
        raise RuntimeError("How peak is not in subgraph keys")

    if peak not in nodes:
        raise RuntimeError("How peak is not in nodes")

    # TODO: end the search as soon as valleys for list of peaks completed
    max_min_densities = dijkstra_max_min_density(subgraph, peak)

    higherdensity_links = {k:v for k, v in max_min_densities.items() if k in density_peaks[i:]}
    si_indices = {key: 1 - value / densities[peak] \
        for key, value in higherdensity_links.items()}
    # for si_key, si_value in si_indices.items():
    #     if si_value != float('inf') and si_value != float('-inf') and si_value < 0.5:
    #         if otuids[peak] != otuids[si_key]:
    #             print(otuids[peak], otuids[si_key], si_value, otuids[nnbydist_indices[peak]], otuids[nnbydist_indices[si_key]])

    peak_links = [index+i for index, value in enumerate(si_indices.items())\
        if value[1] < 0.5 and value[0] != peak]

    # query_nodes = primary_clusters[i]
    # query_links = pair_links[np.any(np.isin(pair_links, query_nodes), axis=1)]
    # link_counts = []
    # for target_peak in peak_links:
    #     target_nodes = primary_clusters[target_peak]
    #     query_target_links = query_links[np.any(np.isin(query_links, target_nodes), axis=1)]
    #     link_counts.append(query_peaks_indices.shape[0])
    #     print(otuids[peak], otuids[target_peak], query_target_links.shape[0])
    return peak_links

def component(graph, nnbydist_indices, densities, density_peaks, contig_length, otuids, primary_clusters):
    """ get components """

    print('going to pool for Dijkstras algorithm')
    s= time.time()
    pool = Pool(NCPUS-2)

    partial_process_peak = [(i, latent, graph, densities, \
        density_peaks, contig_length, primary_clusters, nnbydist_indices)\
        for i in range(len(density_peaks))]

    print(density_peaks, 'density_peaks')
    peak_links = pool.starmap(process_peak, partial_process_peak)

    pool.close()
    pool.join()

    connected_links = find_connected_components(peak_links)

    total_clusters = len(set(connected_links))
    final_clusters = []
    for c in range(total_clusters):
        indices = np.nonzero(connected_links==c)[0]

        temp_cluster = []
        for idx in indices:
            temp_cluster.extend(primary_clusters[idx])
        
        final_clusters.append(temp_cluster)

    gc.collect()

    with open('clusters', 'w', encoding='utf8') as file1:
        for counter, f in enumerate(final_clusters):
            # print(np.sum(contig_length[f]), counter, 'cluster')
            for q in f:
                file1.write(str(q) + ' ' + contig_names[q] \
                + ' ' + otuids[q] + ' ' + str(counter) + '\n')
    file1.close()

    print(time.time() - s ,'seconds to complete Dijkstras algorithm')
    # return peak_links


def cluster(latent, contig_length):
    """ cluster contigs """
    nnbydist_indices, distance_percontig, labels, \
    distances, densities, graph = get_neighbors(latent, contig_length)
    # densities, density_peaks, nearest, primary_clusters = \
    # get_density_peaks(nnbydist_indices, labels, contig_length, densities)
    # component(graph, nnbydist_indices, densities, \
    #     density_peaks, contig_length, otuids, primary_clusters)

    # return peak_links

if __name__ == "__main__":

    variable = '/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/new_mcdevol_run/vae_byol/' # './'
    # latent = np.load(variable + 'latent_mu.npy')
    latent = np.load('latent_umap.npy')
    contig_length = np.load(variable + '../contigs_2klength.npz',allow_pickle=True)['arr_0']
    contig_names = np.load(variable + '../contigs_2knames.npz',allow_pickle=True)['arr_0']
    otuids = np.loadtxt(variable + '../otuids', dtype='object') # type: ignore

    TOTAL_CONTIGS, LATENT_DIMENSION = latent.shape
    data_indices = np.arange(TOTAL_CONTIGS)
    ETA = 2 / LATENT_DIMENSION

    # # subset
    # selected_indices = pd.read_csv(\
    # '/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/new_mcdevol_run/selected_indices_otu1000counts', header=None)
    # selected_indices = selected_indices[0].to_numpy()
    # latent = latent[selected_indices]
    # TOTAL_CONTIGS, LATENT_DIMENSION = latent.shape
    # data_indices = np.arange(TOTAL_CONTIGS)
    # contig_length = contig_length[selected_indices]
    # otuids = otuids[selected_indices]
    # contig_names = contig_names[selected_indices]
    # latent = np.load('latent_umap_subset.npy', allow_pickle=True)
    # TOTAL_CONTIGS, LATENT_DIMENSION = latent.shape
    # print(latent.shape)

    # pair_links = np.load(variable + '../links_selected.npy')
    cluster(latent, contig_length) # type: ignore

    gc.collect()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                          locals().items())), key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

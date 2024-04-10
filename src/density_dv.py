#!/usr/bin/env python

import os
import heapq
import numpy as np
import hnswlib
import time
import matplotlib.pyplot as plt
from connected_components import group_indices, dijkstra_max_min_density, find_connected_components, density_links
from collections import Counter
from multiprocessing import Pool
import gc
import plotly.express as px

# local memory check
import sys
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
NN_SIZE_CUTOFF = 200000
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
    s= time.time()
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
    nnbydist_indices = [[] for _ in data_indices]
    for i, inds in enumerate(dist_cutoff_indices):
        nn_inds = labels[i,:inds+1]
        # long contig of at least nnsize, or 
        # contig has neighbor list with nnsize beyond distance cutoff
        # restrict neighbor list 
        if len(nn_inds) < 4 or distance_percontig[i] > DISTANCE_CUTOFF:
            idx = np.nonzero(distances[i] <= DISTANCE_CUTOFF)[0]
            nnbydist_indices[i] = labels[i][idx]
            distance_percontig[i] = DISTANCE_CUTOFF
        elif contig_length[i] < 8000:
            # contigs with many neighbors but short and hence noisely include many neighbors
            nnbydist_indices[i] = labels[i][:10]
            distance_percontig[i] = distances[i,10]
        else:
            nnbydist_indices[i] = nn_inds
    gc.collect()

    print(time.time()- s, 'seconds for get neighbors')
   
    return nnbydist_indices, distance_percontig, labels, distances

def get_density_peaks(nnbydist_indices, distance_percontig, labels, distances, length):
    """ compute density using k-nearest neighbors """
    s = time.time()
    
    # densities = np.array([np.sum(length[nnbydist_indices[c]] ** ETA) / distance_percontig[c] \
    #                 for c in range(TOTAL_CONTIGS)])

    # if short contig is wrongly associated with long contig, it gets higher density 
    # densities = np.array([np.sum(nnbydist_indices[c] ** ETA) / distance_percontig[c] \
    #                 for c in range(TOTAL_CONTIGS)])
    
    densities = np.array([length[c] ** ETA / distance_percontig[c] \
                    for c in data_indices])

    graph = {c : list(zip(nnbydist_indices[c][1:], densities[nnbydist_indices[c][1:]]))
            if len(nnbydist_indices[c][1:]) > 0 \
            else [] for c in data_indices}
   
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
    # max_count_nn = max(range(len(nnbydist_indices)), \
    #     key=lambda i: len(nnbydist_indices[i]))
    # print(max_count_nn, 'index of largest nn point')
    # maxnn_otuid = Counter(otuids[nnbydist_indices[max_count_nn]])
    # print('maximum nns having contig otulists: ', end='\t')
    # print(maxnn_otuid)

    nearest = np.full(TOTAL_CONTIGS,-1) # np.arange(TOTAL_CONTIGS)
    density_peaks_flag = np.full(TOTAL_CONTIGS,-1)

    indices = np.argsort(length)[::-1]
    hrd_pointlist = []
    hrd_pointlist1 = []
    for c in indices:
        if nearest[c] < 0:
            neighbor_inds = nnbydist_indices[c]
            maxdensity_point = np.argmax(densities[neighbor_inds])
            if maxdensity_point == 0:
                density_peaks_flag[c] = c
                nearest[neighbor_inds] = c
                hrd_pointlist1.append(c)

            else:
                # TODO: check if another higher density point is closer than max density point.
                # If so, assign nearest[c] to closer highder density point and set that point as peak
                higher_density_point = neighbor_inds[maxdensity_point]
                nearest[c] = neighbor_inds[maxdensity_point]
                if nearest[higher_density_point] < 0:
                    density_peaks_flag[higher_density_point] = higher_density_point
                    hrd_pointlist.append(neighbor_inds[maxdensity_point])
                    nearest[higher_density_point] = higher_density_point

    density_peaks_inds = np.nonzero(density_peaks_flag>=0)[0]

    nearest_prev = np.full(nearest.size, -1, dtype=int)
    while (nearest != nearest_prev).any():
        nearest_prev = nearest
        nearest = nearest[nearest[nearest[nearest[:]]]]

    # Sometimes tracing back through nearest of nearest
    # end up linking one density peak to another
    # Retain only density peaks that are in the tracedback nearest 
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

    print(time.time() - s, len(density_peaks), 'total density peaks', 'seconds for graph and densities calculation')
    gc.collect()

    return densities, density_peaks, graph, nearest


from multiprocessing import Pool
from functools import partial

def process_peak(peak, graph, densities, density_peaks, otuids,\
            nnbydist_indices, separate_peaks, contig_length):
    
    max_min_densities = dijkstra_max_min_density(graph, peak)

    # TODO: check if cutoff for density valley is needed to merge peaks.
    # SI index would help
    higherdensity_links = {k:v for k, v in max_min_densities.items() \
        if (k in density_peaks) and (k != peak) and (v != float('-inf'))}
    # Higher density condition doesn't work when v and densities[peak] is same value
    if not higherdensity_links:
        separate_peaks.add(peak)
    else:
        si_indices = {key: 1 - value / densities[peak] \
            for key, value in higherdensity_links.items()}
        if min(si_indices.values()) > 0.2: # type: ignore
            separate_peaks.add(peak)
        elif contig_length[peak] > 1E6:
            separate_peaks.add(peak)
        else:
            links = dict(filter(lambda item: item[1] < 0.2, si_indices.items()))
            print(len(links), otuids[links.keys()], 'length of links')
            print(links.values())

def component(graph, nnbydist_indices, densities, density_peaks, contig_length):
    """ get components """
    separate_peaks = set()
    components = group_indices(nnbydist_indices)
    connected_indices = [peaks[0] for peaks in components if len(peaks[0]) > 1 ]
    # for i in connected_indices:
#     connected_component = components[i]
#     peaks_incomponent = density_peaks[connected_component[0]]
#     nodes = set(connected_component[1])  # Convert nodes to set for faster membership testing

#     subgraph = {}        
#     subgraph.update({peak: graph[peak] for peak in peaks_incomponent})
    
#     for node in nodes:
#         if node in graph:
#             # subgraph[node] = graph[node]
#             node_edges = graph[node]
#             filtered_edges = [(edge, weight) for edge, weight in node_edges if edge in nodes]
#             if filtered_edges:
#                 subgraph[node] = filtered_edges
    pool = Pool(NCPUS)
    partial_process_peak = partial(process_peak, graph=graph, densities=densities,\
                                density_peaks=density_peaks,\
                                otuids=otuids, nnbydist_indices=nnbydist_indices,\
                                separate_peaks=separate_peaks, contig_length=contig_length)
    pool.map(partial_process_peak, density_peaks)

    pool.close()
    pool.join()

    gc.collect()
    # return peaks


def assign_points(density_peaks, components, connected_links, nnbydist_indices, nearest):
    """ assign points to density peaks and merge closer density peaks """

    cluster_id = np.full(TOTAL_CONTIGS, -1, dtype=int)
    cluster_centers = set()
    for i in components:
        inds, ids = i
        peaks = density_peaks[inds]

        if len(peaks) == 1:
            cluster_id[ids] = peaks[0]
            cluster_centers.add(peaks[0]) 

        else:
            # links format:(peak_indices, peakid_towhich_domerging)
            links = connected_links.pop(0)

            if links[1].size == 0:
                neighbor_ids = []
                for p in peaks:
                    neighbor_ids.extend(nnbydist_indices[p])
                    cluster_id[nnbydist_indices[p]] = p
                    cluster_centers.add(p)

                # Handle common indices         
                counts = Counter(neighbor_ids)
                common_inds = [i for i, c in counts.items() if c > 1]
                for c_id in common_inds:
                    cluster_id[c_id] = nearest[c_id]

            # TODO: As of now, we merge all peaks.
            # However, there can be multiple sub merging based on indices in merge sets
            else:
                cluster_id[ids] = peaks[0]
                cluster_centers.add(peaks[0]) 

    # print(len(cluster_centers), 'cluster centers')


    # for f in unassigned_points:
    #     if len(nnbydist_indices[f]) > 1:
    #         cluster_id[f] = cluster_id[nnbydist_indices[f][1]]
    #     else:
    #         if cluster_id[nearest[f]] != -1:
    #             cluster_id[f] = cluster_id[nearest[f]]
    #         # print(f, cluster_id[nearest[f]], 'unassigned no nearby points')
    # unassigned_points = np.nonzero(cluster_id==-1)[0]
    # count = Counter(cluster_id)
    # for f in count.items():
    #     print(f)

    gc.collect()

def cluster(latent, contig_length):
    """ cluster contigs """
    nnbydist_indices, distance_percontig, labels, distances = get_neighbors(latent, contig_length)
    densities, density_peaks, graph, nearest = get_density_peaks(nnbydist_indices, distance_percontig, labels, distances, contig_length)
    component(graph, nnbydist_indices, densities, density_peaks, contig_length)


if __name__ == "__main__":
  
    variable = '/big/work/mcdevol/cami2_datasets/marine/pooled_assembly/new_mcdevol_run/vae_byol/'
    latent = np.load(variable + 'latent_mu.npy')
    contig_length = np.load(variable + '../contigs_2klength.npz',allow_pickle=True)['arr_0']
    # contig_names = np.load(variable + 'contigs_2knames.npz',allow_pickle=True)['arr_0']
    otuids = np.loadtxt(variable + '../otuids', dtype='object') # type: ignore
    
    TOTAL_CONTIGS, LATENT_DIMENSION = latent.shape
    data_indices = np.arange(TOTAL_CONTIGS)
    ETA = 0.4 # 1 / LATENT_DIMENSION

    cluster(latent, contig_length) # type: ignore
    # points_thatmay_unassigned, points_thatmay_unassigned1, points_thatmay_unassigned2, \
    #     points_thatmay_unassigned3, points_thatmay_unassigned4, false_peak_nns = cluster(latent, contig_length) # type: ignore
    gc.collect()
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                          locals().items())), key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    # labels = np.zeros(TOTAL_CONTIGS,dtype=int)

    # counter = 1
    # for i in range(len(clusters)):
    #     labels[clusters[i]] = counter
    #     counter += 1

    # for j in range(len(labels)):
    #     print(labels[j], otuids[j])

    # import seaborn as sns
    # import plotly.express as px
    # fig = px.scatter(x=latent[:,8], y=latent[:,4])
    # fig.update_traces(marker=dict(size=1))
    # fig.show()



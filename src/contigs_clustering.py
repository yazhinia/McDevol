#!/usr/bin/env python3

# import os, sys
# parent_path = os.path.dirname(os.path.dirname(__file__))
# sys.path.insert(0, parent_path)

import numpy as np
import time
import gc
from util.bayesian_distance import compute_dist
from alive_progress import alive_bar



def cluster_by_centroids(cluster_parameters):
    read_counts = cluster_parameters[0]
    Rc_reads = cluster_parameters[1]
    contig_length = cluster_parameters[2]
    total_contigs = cluster_parameters[3]
    dirichlet_prior = cluster_parameters[4]
    dirichlet_prior_persamples = cluster_parameters[5]
    kmer_counts = cluster_parameters[6]
    Rc_kmers = cluster_parameters[7]
    dirichlet_prior_kmers = cluster_parameters[8]
    dirichlet_prior_perkmers = cluster_parameters[9].flatten()
    d0 = cluster_parameters[10]
    tmp_dir = cluster_parameters[11]
    q_read = cluster_parameters[12]
    q_kmer = cluster_parameters[13]
    members = []
    cluster_curr = 0
    cluster_assigned = np.zeros(total_contigs, dtype=int) - 1
    dist_to_assigned = np.zeros(total_contigs, dtype=float) + d0
    print("entering iterative distance_calculation", flush=True)


    print(cluster_assigned.shape[0], "contigs are being clustered", q_read, q_kmer)
    neighbors = np.empty((total_contigs, 0)).tolist()
    s = time.time()
    iterate_ind = np.argsort(Rc_reads)[::-1]
    print("entering iterative distance_calculation", flush=True)
    clustercentroids_list = []
    with alive_bar(total_contigs, title='contigs processed', spinner=False, theme='classic') as bar:
    
        for c in iterate_ind:
        
            if cluster_assigned[c] < 0 :
                distance = compute_dist(c, read_counts, kmer_counts, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers, q_read, q_kmer)
                clustercentroids_list.append(c)

                inds = np.nonzero(distance < dist_to_assigned)[0] # " there could be empty list "
                if len(inds) > 0 :

                    if distance[c] > d0:
                        inds = np.append(inds, c)

                    dist_to_assigned[inds] = distance[inds]
                    cluster_assigned[inds] = cluster_curr

                    for n in np.nonzero(distance < d0)[0]:
                        neighbors[n].append(cluster_curr)

                    cluster_curr += 1
            bar()

    for k in range(cluster_curr):
        if len(np.nonzero(cluster_assigned==k)[0]) > 0:
            members.append(np.nonzero(cluster_assigned==k)[0])

    for c in np.nonzero(cluster_assigned < 0)[0]:
        cluster_assigned[c] = cluster_curr
        members.append([c])
        neighbors[c].append(cluster_curr)
        cluster_curr += 1

    # np.savez(tmp_dir + '/distance_initialclusters_50000c', distance_matrix)
    # np.savetxt(tmp_dir + "/cluster_assigned_q04q02", cluster_assigned, fmt='%d')

    # cluster_length = []

    # for k in members:
    #     if len(k) == 0:
    #         print("check again")
    #     cluster_centroids_read.append(read_counts[k].sum(axis=0))
    #     cluster_centroids_kmer.append(kmer_counts[k].sum(axis=0))
    #     cluster_length.append(sum(contig_length[k]))
    
    # cluster_length = np.array(cluster_length)
    
    # cluster_centroids_read = np.array(cluster_centroids_read)
    # cluster_centroids_kmer = np.array(cluster_centroids_kmer)

    print(f"Obtained {len(members)} clusters from initial clustering".format(len(members)))
    print("Initial clustering took:", time.time() - s,"seconds")

    gc.collect()

    return members, neighbors


def count_numbers_of_sharedcontigs(K, neighbors, min_shared_contigs):
    s = time.time()
    shared = np.zeros([K,K])

    for c in np.arange(len(neighbors)):

        if len(neighbors[c]) >= 2:

            k_ind = np.array(np.meshgrid(np.array(neighbors[c]),np.array(neighbors[c]))).T.reshape(-1,2)
            k_ind = k_ind[k_ind[:,0]!=k_ind[:,1]]

            for i in k_ind:
                shared[i[0],i[1]] += 1
                
    shared = np.array(shared > min_shared_contigs).astype(int)

    links = []

    for k in np.arange(K):
        links.append(list(np.nonzero(shared[k])[0]))

    print("count_numbers_of_sharedcontigs took ", time.time()-s, "seconds")
    return links

def find_connected_components(links):
    s = time.time()
    K = np.shape(links)[0]
    components = np.zeros(K).astype(int) - 1
    component_curr = 0

    for k in np.arange(K):

        if components[k] < 0:

            candidates = links[k]
            components[k] = component_curr

            while len(candidates) != 0:
                l = candidates.pop(-1)

                if components[l] < 0:
                    components[l] = component_curr
                    candidates.extend(links[l])

            component_curr += 1


    if (component_curr != len(set(components))):
        raise Exception("problem with connected component calculations")
        exit(0)

    print("find_connected_components took ", time.time()-s, "seconds")
    num_components = component_curr
    numclust_incomponents = np.unique(components, return_counts=True)[1]

    return components, num_components, numclust_incomponents

def merge_members_by_connnected_components(components, num_components, members):
    K = len(components)
    print(K, len(members))
    clusters = [[] for i in range(num_components)]

    for k in np.arange(K):
        clusters[components[k]].append(members[k])
 
    for i in np.arange(num_components):
        clusters[i] = np.unique(np.concatenate(clusters[i]).ravel())
    return clusters

def cluster_by_connecting_centroids(cluster_parameters):
    print("computing cluster_by_connecting_centroids", flush=True)
    s = time.time()
    min_shared_contigs = 5
    working_dir = cluster_parameters[12]
    members, neighbors = cluster_by_centroids(cluster_parameters)
    print("distance calculation time:", time.time() - s, "seconds")
    links = count_numbers_of_sharedcontigs(len(members), neighbors, min_shared_contigs)
    components, num_components, numclust_incomponents = find_connected_components(links)
    print("number of connected components", num_components)
    clusters = merge_members_by_connnected_components(components, num_components, members)
    print("count_by_connecting_centroids took ", time.time() - s, "seconds")
    return clusters, numclust_incomponents

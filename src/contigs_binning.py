#!/usr/bin/env python3

import time
import gc
import numpy as np
import distance_calculations as dist
import util.metadevol_distance as calc_distance
from alive_progress import alive_bar

def cluster_by_centroids(argv):
    read_counts = argv[0]
    Rc_reads = argv[1]
    total_contigs = argv[3]
    dirichlet_prior = argv[4]
    dirichlet_prior_persamples = argv[5]
    kmer_counts = argv[6]
    Rc_kmers = argv[7]
    dirichlet_prior_kmers = argv[8]
    dirichlet_prior_perkmers = argv[9].flatten()
    d0 = argv[10]
    d1 = argv[11]
    tmp_dir = argv[13]
    q_read = argv[14]
    q_kmer = argv[15]
    members = []
    cluster_curr = 0
    cluster_assigned = np.zeros(total_contigs, dtype=int) - 1
    dist_to_assigned = np.zeros(total_contigs, dtype=float) + d0
    print(cluster_assigned.shape[0], "contigs are being clustered")
    neighbors = np.empty((total_contigs, 0)).tolist()
    s = time.time()
    iterate_ind = np.argsort(Rc_reads)[::-1]
    print("entering iterative distance_calculation", flush=True)
    print(iterate_ind)

    # distance_list = []
    clustercentroids_list = []
    
    with alive_bar(total_contigs, title='contigs processed', spinner=False, theme='classic') as bar:

        for c in iterate_ind:

            if cluster_assigned[c] < 0 :

                # distance = dist.distance(read_counts[c], read_counts, Rc_reads[c], Rc_reads, dirichlet_prior, dirichlet_prior_persamples, 0)
                # distance_bykmers = dist.distance(kmer_counts[c], kmer_counts, Rc_kmers[c], Rc_kmers, dirichlet_prior_perkmers, dirichlet_prior_kmers, 1)
                # distance = distance_byreads + distance_bykmers
                # distance = calc_distance.compute_kmercountdist(c, kmer_counts, Rc_kmers, dirichlet_prior_kmers, dirichlet_prior_perkmers)
                distance = calc_distance.compute_dist(c, read_counts, kmer_counts, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers, q_read, q_kmer)
                # distance = calc_distance.compute_readcountdist(c, read_counts, Rc_reads, dirichlet_prior, dirichlet_prior_persamples)
                inds = np.nonzero(distance <= dist_to_assigned)[0] # " there can be empty list "
                # inds1 = np.nonzero(distance1 <= dist_to_assigned)[0]

                    
                # distance_list.append(distance)

                if len(inds) > 0 :

                    if distance[c] >= d0:

                        inds = np.append(inds, c)

                dist_to_assigned[inds] = distance[inds]
                cluster_assigned[inds] = cluster_curr
            
                for n in np.nonzero(distance < d1)[0]:
                    neighbors[n].append(cluster_curr)
                    
                cluster_curr += 1
                clustercentroids_list.append(c)

            bar()


    for k in range(cluster_curr):
        if len(np.nonzero(cluster_assigned==k)[0]) > 0:
            members.append(np.nonzero(cluster_assigned==k)[0])

    for c in np.nonzero(cluster_assigned < 0)[0]:
        cluster_assigned[c] = cluster_curr
        members.append([c])
        neighbors[c].append(cluster_curr)
        cluster_curr += 1

    # with open(tmp_dir + "/try/length_2500/cluster_assigned_testforold", 'w+') as file:
    #     for f in range(len(members)):
    #         for q in members[f]:
    #             file.write(str(q) + " " + str(f) + "\n")

    np.savetxt(tmp_dir + '/cluster_assigned_oldcluster', cluster_assigned, fmt='%d')

    print("Obtained {} clusters from initial clustering".format(len(members)))
    print("Initial clustering took:", time.time() - s,"seconds")
    gc.collect()
    return members, neighbors


def count_numbers_of_sharedcontigs(K, neighbors, min_shared_contigs):
    s = time.time()
    shared = np.zeros([K,K], dtype=int)
    print("K")
    for c in np.arange(len(neighbors)):

        if len(neighbors[c]) >= 2:
            k_ind = np.array(np.meshgrid(np.array(neighbors[c]),np.array(neighbors[c]))).T.reshape(-1,2)
            
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
    K = len(links)
    components = np.zeros(K, dtype=int) - 1
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
    clusters = [[] for i in range(num_components)]

    for k in np.arange(K):
        clusters[components[k]].append(members[k])
 
    for i in np.arange(num_components):
        clusters[i] = np.unique(np.concatenate(clusters[i]).ravel())
    return clusters

def cluster_by_connecting_centroids(cluster_parameters):

    print("computing cluster_by_connecting_centroids", flush=True)
    s = time.time()
    argv = cluster_parameters
    min_shared_contigs = argv[12]
    tmp_dir = argv[13]
    members, neighbors = cluster_by_centroids(cluster_parameters)
    print("distance calculation time:", time.time() - s, "seconds")
    print(len(members))
    links = count_numbers_of_sharedcontigs(len(members), neighbors, min_shared_contigs)
    components, num_components, numclust_incomponents = find_connected_components(links)
    np.savetxt(tmp_dir + "/components_bothlinkage", np.dstack((np.arange(len(components)), components))[0], fmt='%d')
    print("number of connected components", num_components)
    print("number of clusters in connected components", numclust_incomponents)
    clusters = merge_members_by_connnected_components(components, num_components, members)
    print("count_by_connecting_centroids took ", time.time() - s, "seconds")
    return clusters, numclust_incomponents

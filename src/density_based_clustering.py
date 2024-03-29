import os, sys
parent_path = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_path)

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
    
    cluster_centroids_read = []
    cluster_centroids_kmer = []
    
    print(cluster_assigned.shape[0], "contigs are being clustered")
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

    np.savetxt(tmp_dir + "/cluster_assigned", cluster_assigned, fmt='%d')

    cluster_length = []

    for k in members:
        if len(k) == 0:
            print("check again")
        cluster_centroids_read.append(read_counts[k].sum(axis=0))
        cluster_centroids_kmer.append(kmer_counts[k].sum(axis=0))
        cluster_length.append(sum(contig_length[k]))
    
    cluster_length = np.array(cluster_length)
    
    cluster_centroids_read = np.array(cluster_centroids_read)
    cluster_centroids_kmer = np.array(cluster_centroids_kmer)

    print("Obtained {} clusters from initial clustering".format(len(members)))
    print("Initial clustering took:", time.time() - s,"seconds")

    gc.collect()
    
    return cluster_assigned, members, cluster_centroids_read, cluster_centroids_kmer, cluster_length


def density_based_clustering(cluster_parameters, cluster_centroids_read, cluster_centroids_kmer, cluster_length):
    
    K = cluster_centroids_read.shape[0]
    dirichlet_prior = cluster_parameters[4]
    dirichlet_prior_persamples = cluster_parameters[5]
    dirichlet_prior_kmers = cluster_parameters[8]
    dirichlet_prior_perkmers = cluster_parameters[9].flatten()
    Rc_reads = np.sum(cluster_centroids_read,axis=1)
    Rc_kmers = cluster_centroids_kmer.reshape(-1,64,4).sum(axis=2)
    q_read = cluster_parameters[12]
    q_kmer = cluster_parameters[13]
    density = np.zeros(K, dtype=int)
    d1 = 1.0
    nearest = np.zeros(K, dtype=int)
    separation_dist = np.zeros(K) + 1e30
    
    for k in range(K):
        distance = compute_dist(k, cluster_centroids_read, cluster_centroids_kmer, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers, q_read, q_kmer)
        inds = np.nonzero(distance < d1)[0]
        
        if distance[k] > d1:
            inds = np.append(inds, k)
        
        nearest[k] = k
        density[k] = sum(cluster_length[inds])
        if len(np.nonzero(distance >= 1e30)[0]) > 0:
            raise RuntimeError("distance is far from 1e30")
        
        for k_p in range(k):

            if distance[k_p] < separation_dist[k] and density[k_p] > density[k]:
                nearest[k] = k_p
                separation_dist[k] = distance[k_p]

            elif distance[k_p] < separation_dist[k_p] and density[k_p] < density[k]:
                nearest[k_p] = k
                separation_dist[k_p] = distance[k_p]

    
    # if len(np.nonzero(separation_dist==1e30)[0]) > 0:
    #     print(nearest[np.nonzero(separation_dist==1e30)[0]])
    #     print(separation_dist[np.nonzero(separation_dist==1e30)[0]])
    #     print("only one should be higher separation_distance")
        
    return density, separation_dist, nearest


def obtain_clusters(density, separation_dist, nearest):

    denssep_threshold = 1000
    sep_min = 1.0

    if density.size != separation_dist.size:
        raise RuntimeError(f'density size and separation distance size doesn\'t match')
    
    cluster_centers = np.nonzero((separation_dist > sep_min) & ((density * separation_dist) > denssep_threshold))[0]
    cluster_curr = cluster_centers.size
    nearest[cluster_centers] = cluster_centers

    nearest_prev = np.zeros(nearest.size, dtype=int) - 1

    while (nearest != nearest_prev).any():
        nearest_prev = nearest
        nearest = nearest[nearest[nearest[nearest[:]]]]

    components = []

    for k in range(cluster_curr):

        if len((np.argwhere(nearest == cluster_centers[k])[0])) > 0:
            components.append(np.nonzero(nearest == cluster_centers[k])[0])

        else:
            raise RuntimeWarning("no clusters in cluster_centers[k] is assigned to nearest[k] ")

    return components

def merge_members_by_connnected_components(components, members):
    K = len(components)
    clusters = [[] for i in range(K)]
    numclust_incomponents = []
    for k in np.arange(K):
        numclust_incomponents.append(len(components[k]))
        for c in components[k]:
            clusters[k].extend(members[c])
            
    return clusters
        

def cluster_by_connecting_centroids(cluster_parameters):
    print("computing cluster_by_connecting_centroids", flush=True)
    s = time.time()
    _, members, cluster_centroids_read, cluster_centroids_kmer, cluster_length = cluster_by_centroids(cluster_parameters)
    ss = time.time()
    density, separation_dist, nearest = density_based_clustering(cluster_parameters, cluster_centroids_read, cluster_centroids_kmer, cluster_length)
    print("density based clustering took", time.time() - ss, "seconds")
    components = obtain_clusters(density, separation_dist, nearest)
    # print(len(separation_dist), len(nearest), len(density), "separation dist, nearest, density")
    clusters = merge_members_by_connnected_components(components, members)
    print("count_by_connecting_centroids took", time.time() - s, "seconds")
    return clusters

#!/usr/bin/env python3

import time
import gc
import numpy as np
import distance_calculations as dist
import metadevol_distance as calc_distance
# from tqdm.auto import tqdm
import click, sys

def cluster_by_centroids(argv):
    read_counts = argv[0]
    Rc_reads = argv[1]
    total_contigs = argv[2]
    dirichlet_prior = argv[3]
    dirichlet_prior_persamples = argv[4]
    kmer_counts = argv[5]
    Rc_kmers = argv[6]
    dirichlet_prior_kmers = argv[7]
    dirichlet_prior_perkmers = argv[8].flatten()
    d0 = argv[9]
    d1 = argv[10]
    working_dir = argv[12]
    members_temp = []
    members = []
    cluster_curr = 0
    cluster_assigned = np.zeros(total_contigs, dtype=int) - 1
    dist_to_assigned = np.zeros(total_contigs, dtype=float) + d0
    print(cluster_assigned.shape[0], "contigs are being clustered")
    neighbors = np.empty((total_contigs, 0)).tolist()
    s = time.time()
    iterate_ind = np.argsort(Rc_reads)[::-1]
    print("entering iterative distance_calculation", flush=True)


    # """ create an extandable table """
    # file = working_dir + "distancen0.h5"
    # import os
    # if os.path.isfile(file):
    #     os.remove(file)
    # hdf = h5py.File(file, 'a')
    # dset = hdf.create_dataset(name='data', shape = (0, total_contigs), maxshape = (None, total_contigs), dtype= "float32")
    with click.progressbar(range(total_contigs), width=100, fill_char=">", empty_char=" ", show_pos=True, file=sys.stderr) as bar:
        for c, i in zip(iterate_ind, bar):
        # # for c in np.arange(C):
            if cluster_assigned[c] < 0 :
                # cluster_assigned[c] = cluster_curr
                # Bayes_factor = dist.calc_Bayes_factor(Rc_reads[c], Rc_reads, contig_length[c], contig_length, gamma_shape, gamma_scale)
                distance = dist.distance(read_counts[c], read_counts, Rc_reads[c], Rc_reads, dirichlet_prior_persamples, dirichlet_prior, 0) #, Bayses_factor
                # distance_bykmers = dist.distance(kmer_counts[c], kmer_counts, Rc_kmers[c], Rc_kmers, dirichlet_prior_perkmers, dirichlet_prior_kmers, 1)
                # distance = distance_byreads + distance_bykmers
                # distance = calc_distance.compute_dist(c, read_counts, kmer_counts, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers)
                # distance = calc_distance.compute_readcountdist(c, read_counts, Rc_reads, dirichlet_prior, dirichlet_prior_persamples)
                inds = np.nonzero(distance < dist_to_assigned)[0] # " there could be empty list "
            
                if len(inds) > 0 :

                    if distance[c] >= d0:
        #                 print(np.min(distance), np.argmin(distance), c, "contig index", distance[c], cluster_assigned[c])
                        inds = np.append(inds, c)

                dist_to_assigned[inds] = distance[inds]
                cluster_assigned[inds] = cluster_curr
            
                for n in np.nonzero(distance < d1)[0]:
                    neighbors[n].append(cluster_curr)
                    
                cluster_curr += 1

    for k in range(cluster_curr):
        if len(np.nonzero(cluster_assigned==k)[0]) > 0:
            members.append(np.nonzero(cluster_assigned==k)[0])

    for c in range(total_contigs):
        if cluster_assigned[c] < 0:
            cluster_assigned[c] = cluster_curr
            members.append([c])
            cluster_curr += 1

    # np.savetxt(working_dir + "/cluster_assigned_onlyreadsq8_test", cluster_assigned, fmt='%d')
                # new cluster steps starts
    #             inds = np.nonzero(distance < dist_to_assigned)[0] # "there could be empty list "
    #             if distance[c] > d0:
    #                 print(np.min(distance), np.argmin(distance), c, "contig index", distance[c], cluster_assigned[c])
    #                 inds = np.append(inds, c)
    #             if len(inds) > 0 :
    #                 dist_to_assigned[inds] = distance[inds]
    #                 members_temp.append(inds)
    #                 cluster_assigned[inds] = cluster_curr
    #             else:
    #                 print("empty indices")
    #                 dist_to_assigned[c] = distance[c]
    #                 members_temp.append([c])
    #                 cluster_assigned[c] = cluster_curr
    #             # new cluster steps ends

    #             # old clustering steps starts
    #             # reassign_ind = np.intersect1d(np.nonzero(cluster_assigned < 0)[0], np.nonzero(np.array(distance < d0))[0])
    #             # if len(reassign_ind) != 0: 
    #             #     members.append(reassign_ind)
    #             #     cluster_assigned[reassign_ind] = cluster_curr
    #             # old cluster steps ends

    #             for n in np.nonzero(distance < d1)[0]:
    #                 neighbors[n].append(cluster_curr)
    #             cluster_curr += 1             

    # # new cluster steps starts
    # for k in range(len(members_temp)):
    #     inds = np.intersect1d(members_temp[k],np.nonzero(cluster_assigned==k)[0])
    #     if len(inds) > 0:
    #         members.append(inds)
    # # new cluster steps ends

            # print("entering {}th cluster center".format(cluster_curr))
    with open(working_dir + "/cluster_assigned", 'w+') as file:
        for f in range(len(members)):
            for q in members[f]:
                file.write(str(q) + " " + str(f) + "\n")


    #     """ all-vs-all distance """
    #     N = 1
    #     dset.resize(dset.shape[0]+N, axis=0)
    #     distance = calc_distance.compute_dist(c, read_counts, kmer_counts, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers)
    #     dset[-N:] = distance
    # hdf.close()



    # np.savetxt(working_dir + "/cluster_assigned_onlyreadsq8", cluster_assigned, fmt='%d')
    print("Obtained {} clusters from initial clustering".format(len(members)))
    print("Initial clustering took:", time.time() - s,"seconds")
    gc.collect()
    return members, neighbors


def count_numbers_of_sharedcontigs(K, neighbors, min_shared_contigs):
    s = time.time()
    shared = np.zeros([K,K])
    print("K")
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
    K = len(links)
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
    min_shared_contigs = argv[11]
    working_dir = argv[12]
    members, neighbors = cluster_by_centroids(cluster_parameters)
    print("distance calculation time:", time.time() - s, "seconds")
    print(len(members))
    links = count_numbers_of_sharedcontigs(len(members), neighbors, min_shared_contigs)
    components, num_components, numclust_incomponents = find_connected_components(links)
    np.savetxt(working_dir + "/components_onlyreads", np.dstack((np.arange(len(components)), components))[0], fmt='%d')
    print("number of connected components", num_components)
    clusters = merge_members_by_connnected_components(components, num_components, members)
    print("count_by_connecting_centroids took ", time.time() - s, "seconds")
    return clusters, numclust_incomponents

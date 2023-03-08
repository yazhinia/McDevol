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
    read_counts_source = argv[13]
    kmer_counts_source = argv[14]
    members = []
    cluster_curr = 0
    cluster_assigned = np.zeros(total_contigs, dtype=int) - 1
    dist_to_assigned = np.zeros(total_contigs, dtype=float) + d0
    
    cluster_centroids_read = np.empty((0, read_counts.shape[1]), dtype=np.float32)
    cluster_centroids_kmer = np.empty((0, 256), dtype=np.float32)
    
    print(cluster_assigned.shape[0], "contigs are being clustered")
    neighbors = np.empty((total_contigs, 0)).tolist()
    s = time.time()
    iterate_ind = np.argsort(Rc_reads)[::-1]
    print("entering iterative distance_calculation", flush=True)
    clustercentroids_list = []

    # dist_file =open(working_dir + 'mddistance_onlyreads','ab')

    with click.progressbar(length=total_contigs, width=100, fill_char=">", empty_char=".", show_pos=True, file=sys.stderr, label="contigs clustered") as bar:
    
        for c, i in zip(iterate_ind, bar):
        
            if cluster_assigned[c] < 0 :
                # distance = dist.distance(read_counts[c], read_counts, Rc_reads[c], Rc_reads, dirichlet_prior_persamples, dirichlet_prior, 0)
                distance = calc_distance.compute_readcountdist(c, read_counts, Rc_reads, dirichlet_prior, dirichlet_prior_persamples)
                # distance = calc_distance.compute_dist(c, read_counts, kmer_counts, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers)
                # clustercentroids_list.append(c)
                # np.savetxt(dist_file, distance)
    # np.savez_compressed(working_dir + 'mddistance_onlyreads.npy',np.array(distance_matrix))
    # dist_file.close()
    # np.save(working_dir + 'clustercentroids_list', np.array(clustercentroids_list))
        #         inds = np.nonzero(distance < dist_to_assigned)[0] # " there could be empty list "
                
        #         if len(inds) > 0 :

        #             if distance[c] > d0:
        # #                 print(np.min(distance), np.argmin(distance), c, "contig index", distance[c], cluster_assigned[c])
        #                 inds = np.append(inds, c)

        #             dist_to_assigned[inds] = distance[inds]
        #             cluster_assigned[inds] = cluster_curr

        #             for n in np.nonzero(distance < d1)[0]:
        #                 neighbors[n].append(cluster_curr)

        #             cluster_curr += 1

    # for k in range(cluster_curr):
    #     if len(np.nonzero(cluster_assigned==k)[0]) > 0:
    #         members.append(np.nonzero(cluster_assigned==k)[0])

    # for c in range(total_contigs):
    #     if cluster_assigned[c] < 0:
    #         cluster_assigned[c] = cluster_curr
    #         members.append([c])
    #         cluster_curr += 1

    # np.savetxt(working_dir + "/cluster_assigned_reads", cluster_assigned, fmt='%d')

    # for k in members:
    #     if len(k) == 0:
    #         print("check again")
    #     cluster_centroids_read = np.append(cluster_centroids_read, np.array([read_counts_source[k].sum(axis=0)]), axis=0)
    #     cluster_centroids_kmer = np.append(cluster_centroids_kmer, np.array([kmer_counts_source[k].sum(axis=0)]), axis=0)

    # print(cluster_curr, len(members))
    # print(np.shape(cluster_centroids_read), np.shape(cluster_centroids_kmer))
    # print("Obtained {} clusters from initial clustering".format(len(members)))
    # print("Initial clustering took:", time.time() - s,"seconds")
    gc.collect()
    return members, cluster_centroids_read, cluster_centroids_kmer


def components_by_merging_clusters(cluster_parameters, members, cluster_centroids_read, cluster_centroids_kmer):
    dirichlet_prior = cluster_parameters[3]
    dirichlet_prior_persamples = cluster_parameters[4]
    dirichlet_prior_kmers = cluster_parameters[7]
    dirichlet_prior_perkmers = cluster_parameters[8].flatten()
    d0 = cluster_parameters[9]
    R_max = cluster_parameters[15]
    Rc_reads = np.sum(cluster_centroids_read, axis=1, keepdims=True)
    cluster_centroids_read_scaled = np.multiply(cluster_centroids_read, R_max/(R_max + Rc_reads)).astype(np.float32)
    Rc_reads = cluster_centroids_read_scaled.sum(axis=1).astype(np.float32)
    
    scale_down_kmer = R_max / (R_max + cluster_centroids_kmer.reshape(-1,64,4).sum(axis=2))
    cluster_centroids_kmer_scaled = np.multiply(cluster_centroids_kmer, np.repeat(scale_down_kmer, 4, axis=1))

    Rc_kmers = cluster_centroids_kmer_scaled.reshape(-1,64,4).sum(axis=2)
    K = len(members)
    print(K, "K in components by merging clusters")
    links = []
    distance_matrix =[]
    for k in range(K):
        distance = dist.distance(cluster_centroids_read_scaled[k], cluster_centroids_read_scaled, Rc_reads[k], Rc_reads, dirichlet_prior_persamples, dirichlet_prior, 0)
        # distance = calc_distance.compute_readcountdist(k, cluster_centroids_read_scaled, Rc_reads, dirichlet_prior, dirichlet_prior_persamples)
        # distance = calc_distance.compute_dist(k, cluster_centroids_read_scaled, cluster_centroids_kmer_scaled, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers)
        distance_matrix.append(distance)
        links.append(list(np.nonzero(distance<d0)[0]))

    np.save(cluster_parameters[12] + "X_readdist_2scpclustercentroids.npy",np.array(distance_matrix))
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
    working_dir = argv[12]
    members, cluster_centroids_read, cluster_centroids_kmer = cluster_by_centroids(cluster_parameters)
    # links = components_by_merging_clusters(cluster_parameters, members, cluster_centroids_read, cluster_centroids_kmer)
    # components, num_components, numclust_incomponents = find_connected_components(links)
    # np.savetxt(working_dir + "/components_reads", np.dstack((np.arange(len(components)), components))[0], fmt='%d')
    # clusters = merge_members_by_connnected_components(components, num_components, members)
    # print("count_by_connecting_centroids took ", time.time() - s, "seconds")
    # return clusters, numclust_incomponents

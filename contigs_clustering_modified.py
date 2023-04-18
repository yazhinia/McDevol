import time
import gc
import numpy as np
import distance_calculations as dist
import metadevol_distance as calc_distance
import optimize_parameters as opt
from alive_progress import alive_bar

def cluster_by_centroids(argv):
    read_counts = argv[0]
    Rc_reads = argv[1]
    contig_length = argv[2]
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
    print(tmp_dir, "tmp_dir")
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

    q_read = np.exp(-8)
    q_kmer = np.exp(-8)
    # dist_file =open(tmp_dir + 'mddistance_onlyreads','ab')

    with alive_bar(total_contigs, title='contigs processed', spinner=False, theme='classic') as bar:
    
        for c in iterate_ind:
        
            if cluster_assigned[c] < 0 :
                distance = calc_distance.compute_dist(c, read_counts, kmer_counts, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers, np.exp(-8.0), np.exp(-8.0))
                clustercentroids_list.append(c)

                inds = np.nonzero(distance < dist_to_assigned)[0] # " there could be empty list "
                
                if len(inds) > 0 :

                    if distance[c] > d0:
                        inds = np.append(inds, c)

                    dist_to_assigned[inds] = distance[inds]
                    cluster_assigned[inds] = cluster_curr

                    for n in np.nonzero(distance < d1)[0]:
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

    np.savetxt(tmp_dir + "/cluster_assigned_summed", cluster_assigned, fmt='%d')

    for k in members:
        if len(k) == 0:
            print("check again")
        cluster_centroids_read.append(read_counts[k].sum(axis=0))
        cluster_centroids_kmer.append(kmer_counts[k].sum(axis=0))
    
    cluster_centroids_read = np.array(cluster_centroids_read)
    cluster_centroids_kmer = np.array(cluster_centroids_kmer)

    # with open(tmp_dir + "/try/length_2500/cluster_assigned_checkcc", 'w+') as file:
    #     for f in range(len(members)):
    #         for q in members[f]:
    #             file.write(str(q) + " " + str(f) + "\n")    

    print(cluster_curr, len(members))

    print("Obtained {} clusters from initial clustering".format(len(members)))
    print("Initial clustering took:", time.time() - s,"seconds")
    # np.save(tmp_dir + 'try/length_2500/cluster_centroids_both', cluster_centroids_read)
    gc.collect()
    return members, cluster_centroids_read, cluster_centroids_kmer


def components_by_merging_clusters(cluster_parameters, members, cluster_centroids_read, cluster_centroids_kmer):
    dirichlet_prior = cluster_parameters[4]
    dirichlet_prior_persamples = cluster_parameters[5]
    dirichlet_prior_kmers = cluster_parameters[8]
    dirichlet_prior_perkmers = cluster_parameters[9].flatten()
    d0 = cluster_parameters[10]
    Rc_reads = np.sum(cluster_centroids_read,axis=1)
    Rc_kmers = cluster_centroids_kmer.reshape(-1,64,4).sum(axis=2)

    K = len(members)
    links = []

    distance_list = []
    for k in range(K):
        inds = []
        # print(cluster_centroids_read[k])
        # distance = dist.distance_connected_component(cluster_centroids_read[k], cluster_centroids_read, Rc_reads[k], Rc_reads, dirichlet_prior_persamples, dirichlet_prior, 0)
        # distance = calc_distance.compute_readcountdist(k, cluster_centroids_read, Rc_reads, dirichlet_prior, dirichlet_prior_persamples)
        # distance = calc_distance.compute_dist(k, cluster_centroids_read, Rc_reads, dirichlet_prior, dirichlet_prior_persamples)
        distance = calc_distance.compute_dist(k, cluster_centroids_read, cluster_centroids_kmer, Rc_reads, Rc_kmers, dirichlet_prior, dirichlet_prior_persamples, dirichlet_prior_kmers, dirichlet_prior_perkmers, np.exp(-8.0), np.exp(-8.0))
        
        inds = list(np.nonzero(distance<d0)[0])
        
        if distance[k] > d0:
            inds.append(k)

        distance_list.append(distance)

        links.append(inds)

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

    print(components, "printing components")

    numclust_incomponents = np.unique(components, return_counts=True)[1]
    print(numclust_incomponents)
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
    tmp_dir = argv[13]
    members, cluster_centroids_read, cluster_centroids_kmer = cluster_by_centroids(cluster_parameters)
    ss = time.time()
    links = components_by_merging_clusters(cluster_parameters, members, cluster_centroids_read, cluster_centroids_kmer)
    print("connected_component using summed count took", time.time()-ss, "seconds")
    components, num_components, numclust_incomponents = find_connected_components(links)
    np.savetxt(tmp_dir + "/components", np.dstack((np.arange(len(components)), components))[0], fmt='%d')
    np.savetxt(tmp_dir + "/numclust", numclust_incomponents, fmt='%d')
    clusters = merge_members_by_connnected_components(components, num_components, members)
    print("number of connected components", num_components)
    print("count_by_connecting_centroids took ", time.time() - s, "seconds")
    
    return clusters, numclust_incomponents

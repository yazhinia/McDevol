__doc__ = """ Utility functions for clustering """
import heapq
import numpy as np

def find(x, parent):
    """ find root of neighbor x """

    if parent[x] != x:
        parent[x] = find(parent[x], parent)
    return parent[x]

def union(x, y, parent, rank):
    """ find union between peak neighbors lists """

    rootx = find(x, parent)
    rooty = find(y, parent)
    if rootx != rooty:
        if rank[rootx] > rank[rooty]:
            parent[rooty] = rootx
        elif rank[rootx] < rank[rooty]:
            parent[rootx] = rooty
        else:
            parent[rooty] = rootx
            rank[rootx] += 1

def group_indices(valid_indices, density_peaks):
    """ group density peaks into connected components """

    peak_neighbors = [set(valid_indices[i]) for i in density_peaks]
    parent = {}
    rank = {}

    for sublist in peak_neighbors:
        for index in sublist:
            parent.setdefault(index, index)
            rank.setdefault(index, 0)

    for sublist in peak_neighbors:
        first = next(iter(sublist))
        for index in sublist:
            union(first, index, parent, rank)

    # merging of indices
    groups = {}
    for index in parent:
        root = find(index, parent)
        groups.setdefault(root, []).append(index)

    index_to_peaks = {}
    for i, sublist in enumerate(peak_neighbors):
        for index in sublist:
            index_to_peaks.setdefault(index, []).append(i)

    result = []
    for group in groups.values():
        merged_sublists = set()
        for index in group:
            merged_sublists.update(index_to_peaks.get(index, []))
        result.append((list(merged_sublists), group))

    return result

def dijkstra_max_min_density(graph, start):
    """ return max density path between start and nearest neighbors """    

    # Initialize maximum minimum densities and priority queue
    max_min_densities = {node: float('-inf') for node in graph}
    max_min_densities[start] = float('inf')
    # Priority queue of (negative density, node)
    priority_queue = [(-max_min_densities[start], start)]

    while priority_queue:
        # Extract node with highest maximum minimum density
        current_density, current_node = heapq.heappop(priority_queue)
        current_density = -current_density

        for neighbor, density in graph[current_node]:
            # Calculate new maximum minimum density for neighbor
            new_density = min(current_density, density)
            if new_density > max_min_densities[neighbor]:
                max_min_densities[neighbor] = new_density

                heapq.heappush(priority_queue, (-new_density, neighbor))

    return max_min_densities

def find_connected_components(merge_links):
    """ find connected components of density peaks """
    peaks = len(merge_links)
    merge_sets = np.full(peaks, -1).astype(int)
    merge_curr = 0
    for k in np.arange(peaks):

        if merge_sets[k] < 0:

            candidates = merge_links[k]

            merge_sets[k] = merge_curr

            while len(candidates) != 0:
                l = candidates.pop(-1)

                if merge_sets[l] < 0:
                    merge_sets[l] = merge_curr
                    candidates.extend(merge_links[l])

            merge_curr += 1

    return merge_sets


def density_links(args):
    """ find density link """
    i, peak, graph, density_peaks, densities = args

    max_min_densities = dijkstra_max_min_density(graph, peak)
    higherdensity_links = {k:v for k, v in max_min_densities.items() \
        if (k in density_peaks) and (k != peak) and (v != float('-inf'))}
    if higherdensity_links:
        max_key = max(higherdensity_links, key=higherdensity_links.get) # type: ignore
        separability_index = 1 - (higherdensity_links[max_key] / densities[peak])
        merge_peakinds = np.where(np.isin(density_peaks, list(higherdensity_links.keys())))[0]
        merge_link = [i] + list(merge_peakinds)
    else:
        separability_index = 1
        merge_link = []
    return i, separability_index, merge_link


# unused part
# peak_neighbors = [valid_indices[i].tolist() for i in density_peaks]
# parent = {}
# rank = {}

# for sublist in peak_neighbors:
#     for index in sublist:
#         if index not in parent:
#             parent[index] = index
#             rank[index] = 0

# for sublist in peak_neighbors:
#     for i in range(1, len(sublist)):
#         union(sublist[0], sublist[i], parent, rank)

# # merging of indices
# groups = {}
# for index in parent:
#     root = find(index, parent)
#     if root not in groups:
#         groups[root] = []
#     groups[root].append(index)
# result = []

# # get indices of peaks being merged
# for group in groups.values():
#     merged_sublists = [i for i, sublist in \
#         enumerate(peak_neighbors) if \
#         any(index in sublist for index in group)]
#     result.append((merged_sublists, group))
# return result
def find(x, parent):
    if parent[x] != x:
        parent[x] = find(parent[x], parent)
    return parent[x]

def union(x, y, parent, rank):
    rootX = find(x, parent)
    rootY = find(y, parent)
    if rootX != rootY:
        if rank[rootX] > rank[rootY]:
            parent[rootY] = rootX
        elif rank[rootX] < rank[rootY]:
            parent[rootX] = rootY
        else:
            parent[rootY] = rootX
            rank[rootX] += 1

def group_indices(list_of_lists):
    parent = {}
    rank = {}
    for sublist in list_of_lists:
        for index in sublist:
            if index not in parent:
                parent[index] = index
                rank[index] = 0
    print('union of sets')
    for sublist in list_of_lists:
        for i in range(1, len(sublist)):
            union(sublist[0], sublist[i], parent, rank)
    
    # merging of indices
    groups = {}
    for index in parent:
        root = find(index, parent)
        if root not in groups:
            groups[root] = []
        groups[root].append(index)
    result = []
    
    # get indices of peaks being merged 
    for group in groups.values():
        merged_sublists = [i for i, sublist in enumerate(list_of_lists) if any(index in sublist for index in group)]
        result.append((merged_sublists, group))
    return result


# def depth_firstsearch(graph, start, visited):
#     visited.add(start)
#     for neighbor, _ in graph[start]:
#         if neighbor not in visited:
#             depth_firstsearch(graph, neighbor, visited)

# def select_subgraph(graph, nodes):
#     visited = set()
#     for node in nodes:
#         if node not in visited:
#             depth_firstsearch(graph, node, visited)
#     subgraph = {node: graph[node] for node in visited}
#     return subgraph
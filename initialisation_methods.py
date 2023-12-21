import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt


from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye 
from random import randint
from sklearn.cluster import KMeans

import networkx as nx
import numpy as np
from networkx.algorithms import community as nx_community



def spectral_clustering(G, k):
    """
    :type G : networkX graph
    :type k : int
    :rtype : dict -> the algorithm must return a dictionary keyed by node to the cluster to which the node belongs
    """
    A = nx.adjacency_matrix(G)
    diagonals = [1/G.degree[node] for node in G.nodes()]
    D_inv = diags(diagonals)
    L = eye(G.number_of_nodes()) - D_inv @ A
    # L = nx.laplacian_matrix(G).astype('float')
    
    # d = k in general but can be equal to other values
    val , vec_eigs = eigs(L, k, which='SM')
    val , vec_eigs = eigs(L, k, which='SR')
    # for i in range(len(val)+1):
    #     print(val[i].real)
    kmeans = KMeans(n_clusters=k)
    # vec_eigs = vec_eigs.real.T
    
    clusters = kmeans.fit_predict(vec_eigs.real)
    i = 0
    tau = np.zeros((len(diagonals),k))
    for node in G.nodes():
        tau[i, clusters[i]] = 1
        i += 1
    
    return tau

def calculate_distance_matrix(G):
    A= nx.adjacency_matrix(G).toarray()
    n_vertices=G.number_of_nodes()
    distance_matrix = np.zeros((n_vertices, n_vertices))


    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            A_copy=A.copy()
            euclidean_matrix = (A_copy - A_copy[i,j])**2
            distance_matrix[i, j] = distance_matrix[j, i] = np.sqrt(euclidean_matrix.sum(axis=1)-A_copy[i,j]**2)[i]
    
    return distance_matrix

def find_closest_clusters(distance_matrix):
    np.fill_diagonal(distance_matrix, np.inf)
    return np.unravel_index(np.argmin(distance_matrix, axis=None), distance_matrix.shape)

def hierarchical_clustering(G, n_clusters):
    n_vertices = G.number_of_nodes()
    clusters = [[i] for i in range(n_vertices)]
    distance_matrix = calculate_distance_matrix(G)

    while len(clusters) > n_clusters:
        i, j = find_closest_clusters(distance_matrix)

        new_cluster = clusters[i] + clusters[j]
        clusters.append(new_cluster)

        clusters.pop(max(i, j))
        clusters.pop(min(i, j))

        new_distances = []
        for k in range(len(distance_matrix)):
            if k != i and k != j:
                new_dist = min(distance_matrix[k, i], distance_matrix[k, j])
                new_distances.append(new_dist)
        new_distances.append(0.0)  

        distance_matrix = np.delete(distance_matrix, [max(i, j), min(i, j)], axis=0)
        distance_matrix = np.delete(distance_matrix, [max(i, j), min(i, j)], axis=1)
        new_distances = np.array(new_distances)
        distance_matrix = np.vstack((distance_matrix, new_distances[:-1]))
        distance_matrix = np.column_stack((distance_matrix, new_distances))
        tau = np.zeros((n_vertices,n_clusters))
    print('clusters', clusters)
    for index, sublist in enumerate(clusters):
        for l in sublist:
            tau[l, index]=1
    return tau

def modularity(G, clustering):
    """
    G (graph)
    clustering (list): n_vertices the index corresponds to the vertex, the value to the cluster
    """
    
    
    modularity=0
    m=G.number_of_edges()
    clustering_copy=clustering.copy()
    for cluster in clustering_copy:
        degree_sum=0

        #cluster=[str(index) for index in cluster]


        sub=G.subgraph(cluster)

        degree_sum = sum([deg for _, deg in sub.degree]) 
        cluster_nedges=sub.number_of_edges()
        modularity+=(cluster_nedges)/m - (degree_sum/(2*m))**2
        
        
    
    return modularity

def all_possible_community_pairs(communities):
    seen_pairs = set()
    for index, community in enumerate(communities):
        for other_community in communities[index+1:]:
                community_pair = tuple((community, other_community))
                if community_pair not in seen_pairs:
                    seen_pairs.add(community_pair)
                    yield community_pair

def best_modularity_change(G, clusters):
    deltas_q={}
    m=G.number_of_edges()
    for i, j in all_possible_community_pairs([i for i in range(len(clusters))]):

        
        clusters_copy=clusters.copy()
        new_cluster = clusters_copy[i] + clusters_copy[j]
        clusters_copy.append(new_cluster)
        clusters_copy.pop(max(i, j))
        clusters_copy.pop(min(i, j))
        delta_q=modularity(G,clusters_copy) - modularity(G, clusters)
        deltas_q[(i,j)]=delta_q
        
    max_value, i, j = max(deltas_q.values()), max(deltas_q, key=lambda k: deltas_q[k])[0], max(deltas_q, key=lambda k: deltas_q[k])[1]
    return max_value, i, j

    

def modularity_clustering(G,n_clusters):
    n_vertices=G.number_of_nodes()
    clusters = [[i] for i in range(n_vertices)]
    delta_q, i, j = best_modularity_change(G, clusters)
    

    while len(clusters)>n_clusters:
        new_cluster = clusters[i] + clusters[j]
        clusters.append(new_cluster)

        clusters.pop(max(i, j))
        clusters.pop(min(i, j))

        delta_q, i, j = best_modularity_change(G, clusters)
 

    tau = np.zeros((n_vertices,len(clusters)))
    for index, sublist in enumerate(clusters):
        for l in sublist:
            tau[l, index]=1
    return tau

def modularity_clustering_optim(G,n_clusters):
    n_vertices=G.number_of_nodes()
    clusters = [[i] for i in range(n_vertices)]
    delta_q, i, j = best_modularity_change(G, clusters)
    

    while len(clusters)>n_clusters:
        new_cluster = clusters[i] + clusters[j]
        clusters.append(new_cluster)

        clusters.pop(max(i, j))
        clusters.pop(min(i, j))

        delta_q, i, j = best_modularity_change_optim(G, clusters)
 

    tau = np.zeros((n_vertices,len(clusters)))
    for index, sublist in enumerate(clusters):
        for l in sublist:
            tau[l, index]=1
    return tau

def modularity_module(graph, n_classes):    
    communities = list(nx_community.louvain_communities(graph))
    
    n_nodes = graph.number_of_nodes()
    tau = np.zeros((n_nodes, n_classes))

    for comm_index, community in enumerate(communities):
        if comm_index < n_classes:
            for node in community:
                tau[node, comm_index] = 1

    return tau

def best_modularity_change_optim(G, clusters):
    deltas_q={}
    m=G.number_of_edges()
    for i, j in all_possible_community_pairs([i for i in range(len(clusters))]):
        sub_graph1=G.subgraph(clusters[i])
        sub_graph2=G.subgraph(clusters[j])
        sub_graph_union=G.subgraph(clusters[i] + clusters[j])

        delta_q=(sub_graph_union.number_of_edges()/m)-((sub_graph1.number_of_edges()/m) + (sub_graph2.number_of_edges()/m)) 
        + (sub_graph2.number_of_nodes()*sub_graph1.number_of_nodes())/m
        deltas_q[(i,j)]=delta_q
    max_value, i, j = max(deltas_q.values()), max(deltas_q, key=lambda k: deltas_q[k])[0], max(deltas_q, key=lambda k: deltas_q[k])[1]
    return max_value, i, j
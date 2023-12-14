import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt


from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye 
from random import randint
from sklearn.cluster import KMeans

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
    n_vertices=G.number_of_nodes()
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

def return_priors_pi(X, tau):
    """
    We want to have tau_iq = P(Z_iq = 1 | X)
    We also have Z_iq = proba that the i th nodes belongs to the cluster q
    At fixed tau, we maximize J(R_X) and output prior and pi 

    Args:
        X (np array): np.array of the graph, of size (n_vertices x n_vertices)
        tau (np.array): last estimation of the tau of size (n_vertices x n_cluster)

    Returns:
        prior, pi: _description_
    """
    n_nodes, n_cluster = tau.shape
    
    prior = np.mean(tau, axis = 0)
    
    # Assumption: tau is a 2D numpy array of shape (n_nodes, n_cluster)
    # and X is a 2D numpy array of shape (n_nodes, n_nodes)
    
    # Calculate the nominator
    # tau[:, :, np.newaxis] shape is (n_nodes, n_cluster, 1)
    # tau[:, np.newaxis, :] shape is (n_nodes, 1, n_cluster)
    # This operation results in a shape of (n_nodes, n_cluster, n_cluster)
    tau_replicated = np.repeat(tau[:, np.newaxis, :, np.newaxis], n_nodes, axis=1)
    theta = tau_replicated * tau_replicated.transpose((1, 0, 3, 2))
    X_expanded = X[:, :, np.newaxis, np.newaxis]
    nominator = np.sum(theta * X_expanded, axis=(0, 1))

    # Calculate the denominator
    # The shape of the denominator is (n_cluster, n_cluster)
    denominator = np.sum(theta, axis=(0, 1))

    # Calculate pi avoiding division by zero
    # Where denominator is zero, we set pi to zero 
    pi = np.divide(nominator, denominator, out=np.zeros_like(nominator), where=denominator != 0)

    return prior, pi


def appro_tau(tau, graph_edges, pi, priors, eps = 1e-04, max_iter = 50):    
    
    finish = False
    current_iter = 0

    while not finish and current_iter < max_iter:
        
        old_tau = tau.copy()
        # Create index arrays
        exp_term = (pi ** graph_edges[:, :, np.newaxis, np.newaxis]) * ((1 - pi) ** (1 - graph_edges[:, :, np.newaxis, np.newaxis]))
        K = exp_term ** old_tau[np.newaxis, :, np.newaxis, :] 

        K = np.prod(K, axis=3)
        for i in range(K.shape[0]):
            K[i, i, :] = 1

        tau = np.prod(K, axis=1)
        tau = tau * priors

        # Reshape new_tau if necessary to match the original tau shape
        tau = tau.reshape(old_tau.shape)
        
        tau = tau / tau.sum(axis=1, keepdims=True)
        
        difference_matrix = np.abs(tau - old_tau)
        if np.all(difference_matrix < eps):
            finish = True    
        current_iter += 1

    return tau

    
def main(X, n_clusters, max_iter = 100, method = "hierarchical"):
    n_nodes, _ = X.shape

    G = nx.from_numpy_array(X)
    
    # Initialize tau 
    if method == "spectral":
        tau = spectral_clustering(G, n_clusters)
    elif method=='hierarchical':
        tau=hierarchical_clustering(G, n_clusters)
    elif method == "random":
        tau = np.random.uniform(0, 1, size=(n_nodes, n_clusters))
        tau = tau / tau.sum(axis=1, keepdims=True)
 
    finished = False
    current_iter = 0
 
    while current_iter < max_iter and not finished:
        priors, pi = return_priors_pi(X, tau.copy())
        new_tau = appro_tau(tau.copy(), X, pi.copy(), priors.copy())
        # new_tau = approximate_tau_step_by_step(tau.copy(), X, pi.copy(), priors.copy())
        
        if np.any(np.isnan(new_tau)):
            break
        diff = new_tau.copy() - tau.copy()
        
        tau = new_tau.copy()

        current_iter += 1
    return priors, pi

def get_X_from_graph(graph):
    n_nodes = len(graph.nodes)
    graph_edges = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in graph.edges:
        graph_edges[i, j] = 1
        graph_edges[j, i] = 1
        
    return graph_edges
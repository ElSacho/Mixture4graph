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

    
def main(X, n_clusters, max_iter = 100, method = "spectral"):
    n_nodes, _ = X.shape

    G = nx.from_numpy_array(X)
    
    # Initialize tau 
    if method == "spectral":
        tau = spectral_clustering(G, n_clusters)
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
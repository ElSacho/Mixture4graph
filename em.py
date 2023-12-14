import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt

from utils import plot_JRX, plot_ICL

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

def J_R_x(graph_edges, tau, pi, priors):
    # Create index arrays
    n_nodes = graph_edges.shape[0]
    
    J_R_x = 0
    
    tau_log_priors = np.where(priors == 0, 0, tau * np.log(priors))
    sum_tau_log_priors = np.sum(tau_log_priors, axis=(0, 1))
    
    J_R_x += sum_tau_log_priors
    
    exp_term = (pi ** graph_edges[:, :, np.newaxis, np.newaxis]) * ((1 - pi) ** (1 - graph_edges[:, :, np.newaxis, np.newaxis]))
    exp_term = np.where(exp_term == 0, 0, np.log(exp_term)) 
    tau_replicated = np.repeat(tau[:, np.newaxis, :, np.newaxis], n_nodes, axis=1)
    theta = tau_replicated * tau_replicated.transpose((1, 0, 3, 2))
    tau_tau_log_b = theta * exp_term
    for i in range(n_nodes):
        tau_tau_log_b[i, i, :, :] = 0
    sum_tau_tau_log_b = np.sum(tau_tau_log_b, axis=(0, 1, 2, 3)) / 2
    
    J_R_x += sum_tau_tau_log_b
    
    tau_log_tau = np.where(tau == 0, 0, tau * np.log(tau))
    sum_tau_log_tau = np.sum(tau_log_tau, axis=(0,1))
    
    J_R_x += sum_tau_log_tau

    return J_R_x

def log_likehood(graph_edges, tau, pi, priors):
    # From tau we create Z
    max_values = np.max(tau, axis=1, keepdims=True)
    mask = (tau == max_values)
    Z = np.zeros_like(tau)
    Z[mask] = 1
    
    log_likehood = 0
    
    Z_log_priors = np.where(priors == 0, 0, Z * np.log(priors))
    sum_Z_log_priors = np.sum(Z_log_priors, axis=(0, 1))
    
    log_likehood += sum_Z_log_priors
    
    exp_term = (pi ** graph_edges[:, :, np.newaxis, np.newaxis]) * ((1 - pi) ** (1 - graph_edges[:, :, np.newaxis, np.newaxis]))
    exp_term = np.where(exp_term == 0, 0, np.log(exp_term)) 
    Z_replicated = np.repeat(Z[:, np.newaxis, :, np.newaxis], graph_edges.shape[0], axis=1)
    theta = Z_replicated * Z_replicated.transpose((1, 0, 3, 2))
    Z_Z_log_b = theta * exp_term
    for i in range(graph_edges.shape[0]):
        Z_Z_log_b[i, i, :, :] = 0
    sum_Z_Z_log_b = np.sum(Z_Z_log_b, axis=(0, 1, 2, 3)) / 2
    
    log_likehood += sum_Z_Z_log_b

    return log_likehood    

def ICL(graph_edges, tau, pi, priors):
    icl = 0
    
    log_lik = log_likehood(graph_edges, tau, pi, priors)
    
    icl += log_lik
    
    n_nodes, n_clusters = tau.shape
    
    m_Q = -1/4 * (n_clusters + 1) * n_clusters * np.log(n_nodes * (n_nodes- 1 ) / 2) - (n_clusters - 1) / 2 * np.log(n_nodes)
    
    icl += m_Q
    
    return icl
  
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
    tab_jrx = []
 
    while current_iter < max_iter and not finished:
        priors, pi = return_priors_pi(X, tau.copy())
        
        tab_jrx.append(J_R_x(X, tau, pi, priors))
        new_tau = appro_tau(tau.copy(), X, pi.copy(), priors.copy())
        # new_tau = approximate_tau_step_by_step(tau.copy(), X, pi.copy(), priors.copy())
        
        if np.any(np.isnan(new_tau)):
            break
        diff = new_tau.copy() - tau.copy()
        
        tau = new_tau.copy()

        current_iter += 1
    return priors, pi, tau, tab_jrx

def get_X_from_graph(graph):
    n_nodes = len(graph.nodes)
    graph_edges = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in graph.edges:
        graph_edges[i, j] = 1
        graph_edges[j, i] = 1
        
    return graph_edges


class mixtureModel():
    def __init__(self, graph, max_iter_EM = 50, initilisation_method = 'spectral'):
        self.graph = graph
        self.graph_edges = get_X_from_graph(graph)
        self.max_iter = max_iter_EM
        self.initilisation_method = initilisation_method
        self.results = {}
    
    def EM(self, n_clusters, max_iter = None, initilisation_method = None):
        if max_iter == None:
            max_iter = self.max_iter
        if initilisation_method == None:
            initilisation_method = self.initilisation_method
        priors, pi, tau, tab_jrx = main(self.graph_edges, n_clusters, max_iter = max_iter, method = initilisation_method)
        ICL_clusters = ICL(self.graph_edges, tau, pi, priors)
        result = {'pi': pi, 'tau' : tau, 'jrx' : tab_jrx, 'priors' : priors, 'ICL' : ICL_clusters, 'max_iter' : max_iter, 'initialisation' : initilisation_method, 'n_clusters' : n_clusters}
        self.results[n_clusters] = result
        
    def fit_tab_clusters(self, tab_n_clusters, max_iter = None, initilisation_method = None):
        if max_iter == None:
            max_iter = self.max_iter
        if initilisation_method == None:
            initilisation_method = self.initilisation_method
        for n_clusters in tab_n_clusters:
            self.EM(n_clusters, max_iter, initilisation_method)
            
    def plot_jrx(self, tab_n_clusters):
        for n_clusters in tab_n_clusters:
            plot_JRX(self.results[n_clusters]['jrx'])
            
    def plot_icl(self):
        tab_clusters = []
        tab_ICL = []
        for n_clusters in self.results.keys():
            tab_clusters.append(n_clusters)
            tab_ICL.append(self.results[n_clusters]['ICL'])
        plot_ICL(tab_clusters, tab_ICL)
        
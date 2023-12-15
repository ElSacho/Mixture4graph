import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt

from utils import plot_JRX, plot_ICL


from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye 
from random import randint
from sklearn.cluster import KMeans

from initialisation_methods import spectral_clustering, hierarchical_clustering, modularity_clustering

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
    
    tau = sp.csr_matrix(tau)
        
    prior = np.array(tau.mean(axis=0)).ravel()

    # Assumption: tau is a 2D numpy array of shape (n_nodes, n_cluster)
    # and X is a 2D sparse matrix of shape (n_nodes, n_nodes)

    # Calculate the nominator
    # For sparse operations, we avoid replicating tau to a 4D structure
    # Instead, we use broadcasting and matrix operations

    # Create sparse diagonal blocks of tau for each cluster
    tau_sparse_blocks = [sp.diags(tau[:, i]) for i in range(n_cluster)]

    nominator_blocks = []
    for i in range(n_cluster):
        for j in range(n_cluster):
            # Element-wise multiplication of the sparse diagonal blocks and X
            block = tau_sparse_blocks[i] @ X @ tau_sparse_blocks[j]
            nominator_blocks.append(block)

    # Sum over all blocks to get the nominator matrix
    # We stack the blocks and sum along the new dimension
    nominator = sp.bmat(nominator_blocks).sum(axis=0)

    # The shape of the denominator is (n_cluster, n_cluster)
    # We use the same blocks but sum over nodes only
    denominator_blocks = [block.sum(axis=0) for block in nominator_blocks]
    denominator = np.array(denominator_blocks).reshape(n_cluster, n_cluster)

    # Calculate pi avoiding division by zero
    # Where denominator is zero, we set pi to zero 
    pi = np.divide(nominator.toarray(), denominator, out=np.zeros_like(denominator), where=denominator != 0)
    tau = tau.toarray()

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
    
    # tau_log_priors = np.where(priors == 0, 0, tau * np.log(priors))
    # tau_log_priors = np.zeros_like(tau)
    non_zero_indices = priors != 0
    log_priors = np.zeros_like(priors)
    log_priors[non_zero_indices] = np.log(priors[non_zero_indices])
    tau_log_priors = tau * log_priors

    sum_tau_log_priors = np.sum(tau_log_priors, axis=(0, 1))
    
    J_R_x += sum_tau_log_priors
    
    exp_term = (pi ** graph_edges[:, :, np.newaxis, np.newaxis]) * ((1 - pi) ** (1 - graph_edges[:, :, np.newaxis, np.newaxis]))
    exp_term_temp = np.zeros_like(exp_term)
    non_zero_indices = exp_term != 0
    exp_term[non_zero_indices] = np.log(exp_term[non_zero_indices])
    # exp_term = np.where(exp_term == 0, 0, np.log(exp_term)) 
    tau_replicated = np.repeat(tau[:, np.newaxis, :, np.newaxis], n_nodes, axis=1)
    theta = tau_replicated * tau_replicated.transpose((1, 0, 3, 2))
    tau_tau_log_b = theta * exp_term
    for i in range(n_nodes):
        tau_tau_log_b[i, i, :, :] = 0
    sum_tau_tau_log_b = np.sum(tau_tau_log_b, axis=(0, 1, 2, 3)) / 2
    
    J_R_x += sum_tau_tau_log_b
    
    # tau_log_tau = np.where(tau == 0, 0, tau * np.log(tau))
    tau_log_tau = np.zeros_like(tau)
    non_zero_indices = tau != 0
    tau_log_tau[non_zero_indices] = tau[non_zero_indices] * np.log(tau[non_zero_indices])
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
 
def from_tau_to_Z(tau):
    max_values = np.max(tau, axis=1, keepdims=True)
    mask = (tau == max_values)
    z = np.zeros_like(tau)
    z[mask] = 1
    return z
   
def main(X, n_clusters, max_iter = 100, method = "spectral"):
    n_nodes, _ = X.shape

    G = nx.from_numpy_array(X)
    
    # Initialize tau 
    if method == "spectral":
        tau = spectral_clustering(G, n_clusters)
    elif method == "random":
        tau = np.random.uniform(0, 1, size=(n_nodes, n_clusters))
        tau = tau / tau.sum(axis=1, keepdims=True)
    elif method == "hierarchical":
        tau = hierarchical_clustering(G, n_clusters)
    elif method == 'modularity':
        tau = modularity_clustering(G, n_clusters)
        
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
    graph_edges = nx.to_numpy_array(graph)
    #  =
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
        
    def fit(self, tab_n_clusters = [2,3,4,5,6,7,8], n_clusters = None, max_iter = None, initilisation_method = None):
        if max_iter == None:
            max_iter = self.max_iter
        if initilisation_method == None:
            initilisation_method = self.initilisation_method
        if n_clusters == None:
            for n_cluster in tab_n_clusters:
                self.EM(n_cluster, max_iter, initilisation_method)
        else :
            self.EM(n_clusters, max_iter, initilisation_method)
            
    def plot_jrx_several_plot(self, tab_n_clusters):
        for n_clusters in tab_n_clusters:
            plot_JRX(self.results[n_clusters]['jrx'], n_clusters)
    
    def plot_jrx(self):
        for n_clusters in self.results.keys():
            # Tracer le graphique en fonction des indices
            plt.plot(self.results[n_clusters]['jrx'], label=f'{n_clusters} clusters') 

        plt.title(r'$\mathcal{J}(R_{\mathcal{X}})$ values')
        # Ajouter des étiquettes aux axes
        plt.xlabel('Iterations')
        plt.ylabel(r'$\mathcal{J}(R_{\mathcal{X}})$')

        # Ajouter une légende au graphique
        plt.legend()

        # Afficher le graphique
        plt.show()
            
    def plot_icl(self):
        tab_clusters = []
        tab_ICL = []
        for n_clusters in self.results.keys():
            tab_clusters.append(n_clusters)
            tab_ICL.append(self.results[n_clusters]['ICL'])
        plot_ICL(tab_clusters, tab_ICL)
    
    def plot_adjency_matrix(self, n_clusters):
        # Nous allons créer une matrice d'adjacence d'exemple avec des blocs pour simuler les clusters
        z = from_tau_to_Z(self.results[n_clusters]['tau'])
        cluster_indices = {q: np.where(z[:, q] == 1)[0] for q in range(n_clusters)}

        # Permuter la matrice d'adjacence
        adjacency_matrix = nx.to_numpy_array(self.graph)
        new_order = np.concatenate([cluster_indices[q] for q in range(n_clusters)])
        permuted_matrix = adjacency_matrix[np.ix_(new_order, new_order)]
        
        # Visualisation de la matrice d'adjacence triée
        plt.figure(figsize=(6, 6))
        plt.spy(permuted_matrix, markersize=0.5)

        # Ajouter des délimitations entre les clusters
        current_idx = 0
        for q in range(n_clusters):
            cluster_size = len(cluster_indices[q])
            if cluster_size > 0:
                current_idx += cluster_size
                plt.axvline(x=current_idx - 0.5, color='r', linestyle='--')
                plt.axhline(y=current_idx - 0.5, color='r', linestyle='--')

        plt.title("Matrice d'adjacence avec nœuds regroupés par cluster")
        plt.show()


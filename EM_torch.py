import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle

from utils import plot_JRX, plot_ICL
from initialisation_methods import spectral_clustering, hierarchical_clustering, modularity_clustering

def return_priors_pi(graph_edges, tau):
    """
    We want to have tau_iq = P(Z_iq = 1 | X)
    We also have Z_iq = proba that the i th nodes belongs to the cluster q
    At fixed tau, we maximize J(R_X) and output prior and pi 

    Args:
        graph_edges (np array): np.array of the graph, of size (n_vertices, n_vertices)
        tau (np.array): last estimation of the tau of size (n_vertices, n_cluster)

    Returns:
        prior, pi: _description_
    """
    # Assuming tau and graph_edges are PyTorch tensors of appropriate shapes
    n_nodes, n_cluster = tau.shape

    # Calculate the prior
    prior = torch.mean(tau, dim=0)

    # Replicate and transpose tau
    # tau[:, :, None] is equivalent to tau.unsqueeze(2) in PyTorch
    # tau[:, None, :] is equivalent to tau.unsqueeze(1) in PyTorch
    # tau_replicated = tau.unsqueeze(1).repeat(1, n_nodes, 1, 1)
    # theta = tau_replicated * tau_replicated.permute(1, 0, 3, 2)

    # # Expand graph_edges and calculate the nominator
    # X_expanded = X[:, :, None, None]
    # nominator = torch.sum(theta * X_expanded, dim=(0, 1))
    tau_replicated = tau.unsqueeze(1).unsqueeze(-1).repeat(1, n_nodes, 1, 1)
    theta = tau_replicated * tau_replicated.transpose(1, 0).transpose(3, 2)
    graph_edges_expanded = graph_edges.unsqueeze(2).unsqueeze(-1)
    nominator = torch.sum(theta * graph_edges_expanded, dim=(0, 1))

    # Calculate the denominator
    denominator = torch.sum(theta, dim=(0, 1))

    # Calculate pi with safe division # WE NEED TO ADD THE CASE WHERE DENOM == 0
    pi = torch.div(nominator, denominator + torch.finfo(torch.float64).eps, out=torch.zeros_like(nominator))
    zero_indices = denominator == 0
    pi[zero_indices] = 0

    return prior, pi

def appro_tau(tau, graph_edges, pi, priors, eps = 1e-04, max_iter = 50):    
    
    finish = False
    current_iter = 0

    while not finish and current_iter < max_iter:
        old_tau = tau
        # Create index arrays
        exp_term = (pi ** graph_edges[:, :, None, None]) * ((1 - pi) ** (1 - graph_edges[:, :, None, None]))
        K = exp_term ** old_tau[None, :, None, :]

        # Calculate the product along the specified axis
        K = torch.prod(K, dim=3)

        # Set diagonal elements to 1
        eye = torch.eye(K.size(0), dtype=torch.bool, device=K.device)
        K[eye] = 1

        # Calculate product and multiply by priors
        tau = torch.prod(K, dim=1)
        tau = tau * priors

        # Reshape tau to match the original shape and normalize
        tau = tau.view(old_tau.shape)
        tau = tau / tau.sum(dim=1, keepdim=True)

        # Calculate the difference and check convergence
        difference_matrix = torch.abs(tau - old_tau)
        finish = torch.all(difference_matrix < eps)
        current_iter += 1
        
    return tau

def J_R_x(graph_edges, tau, pi, priors):
    # Create index arrays
    n_nodes = graph_edges.shape[0]
    
    J_R_x = 0
    
    non_zero_indices = priors != 0
    log_priors = torch.zeros_like(priors)
    log_priors[non_zero_indices] = torch.log(priors[non_zero_indices])
    tau_log_priors = tau * log_priors

    sum_tau_log_priors = torch.sum(tau_log_priors, dim=(0, 1))
    
    J_R_x += sum_tau_log_priors
    
    exp_term = (pi ** graph_edges[:, :, None, None]) * ((1 - pi) ** (1 - graph_edges[:, :, None, None]))
    non_zero_indices = exp_term != 0
    exp_term_log = torch.zeros_like(exp_term)
    exp_term_log[non_zero_indices] = torch.log(exp_term[non_zero_indices])
    # tau_replicated = tau.unsqueeze(1).repeat(1, n_nodes, 1, 1)
    tau_replicated = tau.unsqueeze(1).unsqueeze(-1).repeat(1, n_nodes, 1, 1)
    theta = tau_replicated * tau_replicated.transpose(1, 0).transpose(3, 2)
    tau_tau_log_b = theta * exp_term_log
    for i in range(n_nodes):
        tau_tau_log_b[i, i, :, :] = 0
    sum_tau_tau_log_b = torch.sum(tau_tau_log_b) / 2
    
    J_R_x += sum_tau_tau_log_b
    
    non_zero_indices = tau != 0
    tau_log_tau = torch.zeros_like(tau)
    tau_log_tau[non_zero_indices] = tau[non_zero_indices] * torch.log(tau[non_zero_indices])
    sum_tau_log_tau = torch.sum(tau_log_tau, dim=(0,1))
    
    J_R_x += sum_tau_log_tau

    return J_R_x.item()

def log_likehood(graph_edges, tau, pi, priors):
    # From tau we create Z
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_nodes = graph_edges.shape[0]
    
    z = from_tau_to_Z(tau)
    
    log_likehood = 0
    
    non_zero_indices = priors != 0
    log_priors = torch.zeros_like(priors).to(device)
    log_priors[non_zero_indices] = torch.log(priors[non_zero_indices])
    z_log_priors = z * log_priors
    sum_z_log_priors = torch.sum(z_log_priors, dim=(0, 1))
        
    log_likehood += sum_z_log_priors
    
    exp_term = (pi ** graph_edges[:, :, None, None]) * ((1 - pi) ** (1 - graph_edges[:, :, None, None]))
    non_zero_indices = exp_term != 0
    exp_term_log = torch.zeros_like(exp_term).to(device)
    exp_term_log[non_zero_indices] = torch.log(exp_term[non_zero_indices])
    z_replicated = z.unsqueeze(1).unsqueeze(-1).repeat(1, n_nodes, 1, 1)
    theta = z_replicated * z_replicated.transpose(1, 0).transpose(3, 2)
    z_z_log_b = theta * exp_term_log
    for i in range(n_nodes):
        z_z_log_b[i, i, :, :] = 0
    sum_z_z_log_b = torch.sum(z_z_log_b) / 2    
    
    log_likehood += sum_z_z_log_b

    return log_likehood.item()

def ICL(graph_edges, tau, pi, priors):
    icl = 0
    
    log_lik = log_likehood(graph_edges, tau, pi, priors)
    
    icl += log_lik
    
    n_nodes, n_clusters = tau.shape
    
    m_Q = -1/4 * (n_clusters + 1) * n_clusters * np.log(n_nodes * (n_nodes- 1 ) / 2) - (n_clusters - 1) / 2 * np.log(n_nodes)
    
    icl += m_Q
    
    return icl
 
def from_tau_to_Z(tau):
    max_values = torch.max(tau, dim=1, keepdim=True)[0]
    mask = (tau == max_values)
    z = torch.zeros_like(tau)
    z[mask] = 1
    return z
   
def main(graph_edges, n_clusters, max_iter = 100, method = "spectral"):
    n_nodes, _ = graph_edges.shape

    G = nx.from_numpy_array(graph_edges.to('cpu').numpy())
    
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
    
    # Define the target device (CPU or CUDA/GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move 'tau' and 'X' tensors to the target device
    tau = torch.from_numpy(tau).to(device)
    graph_edges = torch.Tensor(graph_edges).to(device)
 
    while current_iter < max_iter and not finished:
        priors, pi = return_priors_pi(graph_edges, tau)
        
        tab_jrx.append(J_R_x(graph_edges, tau, pi, priors))
        new_tau = appro_tau(tau, graph_edges, pi, priors)
        # new_tau = approximate_tau_step_by_step(tau.copy(), X, pi.copy(), priors.copy())
        
        if torch.any(torch.isnan(new_tau)):
            break
    
        tau = new_tau

        current_iter += 1
    return priors, pi, tau, tab_jrx

def get_X_from_graph(graph):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_edges = nx.to_numpy_array(graph)
    graph_edges_tensor = torch.Tensor(graph_edges).to(device)
    return graph_edges_tensor
    n_nodes = len(graph.nodes)
    graph_edges = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in graph.edges:
        graph_edges[i, j] = 1
        graph_edges[j, i] = 1
        
    return graph_edges

class mixtureModel():
    def __init__(self, graph, max_iter_EM = 50, initilisation_method = 'spectral'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        result = {'pi': pi.to('cpu').numpy(), 'tau' : tau.to('cpu').numpy(), 'jrx' : tab_jrx, 'priors' : priors.to('cpu').numpy(), 'ICL' : ICL_clusters, 'max_iter' : max_iter, 'initialisation' : initilisation_method, 'n_clusters' : n_clusters}
        self.results[n_clusters] = result
        
    def fit(self, tab_n_clusters = [2,3,4,5,6,7,8], n_clusters = None, max_iter = None, initilisation_method = None, save_path = "save_results.pkl"):
        if max_iter == None:
            max_iter = self.max_iter
        if initilisation_method == None:
            initilisation_method = self.initilisation_method
        if n_clusters == None:
            for n_cluster in tab_n_clusters:
                self.EM(n_cluster, max_iter, initilisation_method)
                print('Fit finished for ', n_cluster, ' clusters ')
        else :
            self.EM(n_clusters, max_iter, initilisation_method)
        with open(save_path+'.pkl', 'wb') as f:
            pickle.dump(self.results, f)
            
    def plot_jrx_several_plot(self, tab_n_clusters):
        for n_clusters in tab_n_clusters:
            plot_JRX(self.results[n_clusters]['jrx'], n_clusters)
    
    def plot_jrx(self, save_path = None):
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
        
        if save_path != None:
            plt.savefig(f'{save_path}.png')
            plt.close()
        else : 
            plt.show()
            
    def plot_icl(self, save_path = None):
        tab_clusters = []
        tab_ICL = []
        for n_clusters in self.results.keys():
            tab_clusters.append(n_clusters)
            tab_ICL.append(self.results[n_clusters]['ICL'])
        plot_ICL(tab_clusters, tab_ICL, save_path)
    
    def plot_adjency_matrix(self, n_clusters, save_path = None):
        # Nous allons créer une matrice d'adjacence d'exemple avec des blocs pour simuler les clusters
        z = from_tau_to_Z(torch.from_numpy(self.results[n_clusters]['tau']))
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
        if save_path != None:
            plt.savefig(f'{save_path}.png')
            plt.close()
        else : 
            plt.show()
            
    def plot_all_adjency_matrices(self, save_path = None):
        for n_clusters in self.results.keys():
            if save_path != None:
                self.plot_adjency_matrix(n_clusters, save_path + "_"+ str(n_clusters)+"_clusters")
            else:
                self.plot_adjency_matrix(n_clusters, save_path = None)
        
    def load_results(self, results_path):
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)
        


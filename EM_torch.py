import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle

from utils import plot_JRX, plot_ICL
from initialisation_methods import spectral_clustering, hierarchical_clustering, modularity_clustering, modularity_module

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

def return_priors_pi_from_graph(graph, tau):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_tau = torch.Tensor(tau).to(device)
    prior, pi = return_priors_pi(get_X_from_graph(graph), tensor_tau)
    return prior.numpy(), pi.numpy()

def appro_tau(tau, graph_edges, pi, priors, eps = 1e-04, max_iter = 50):    
    
    finish = False
    current_iter = 0

    while not finish and current_iter < max_iter:
        old_tau = tau
        # Create index arrays
        eps = torch.finfo(torch.float32).eps
        exp_term = ( (pi+eps) ** graph_edges[:, :, None, None]) * ((1 - pi + eps) ** (1 - graph_edges[:, :, None, None]))
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

def calculated_empirical_clustering_coefficient(priors, pi):
    n_cluster = pi.shape[0]
    num = 0
    den = 0
    for q in range(n_cluster):
        for l in range(n_cluster):
            for m in range(n_cluster):
                num += priors[q]*priors[l]*priors[m] * pi[q,l]* pi[q,m]* pi[m,l]
                den += priors[q]*priors[l]*priors[m] * pi[q,l]* pi[q,m]
    if den != 0:
        return num / den
    return 0
   
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
        tau = modularity_module(G, n_clusters)
        
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
    def __init__(self, graph, max_iter_EM = 50, initilisation_method = 'random', use_GPU = True):
        """_summary_

        Args:
            graph (nx graph): the graph on which you want to perfom your EM algorithm
            max_iter_EM (int, optional): The number of iterations for the EM algorithm. Defaults to 50.
            initilisation_method (str, optional): The initialisation method you want to use. Defaults to 'random'.
            use_GPU (bool, optional): Wheter or not you want to use GPU if you have access to GPU 
            (you may want to use_GPU=False if you have GPU memory issues) Defaults to True.
        """
        if use_GPU :
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            DEVICE = 'cpu'
        self.graph = graph
        self.graph_edges = get_X_from_graph(graph)
        self.max_iter = max_iter_EM
        self.initilisation_method = initilisation_method
        self.results = {}
        self.ICL_values = {}
    
    def EM(self, n_clusters, max_iter = None, initilisation_method = None):
        if max_iter == None:
            max_iter = self.max_iter
        if initilisation_method == None:
            initilisation_method = self.initilisation_method
        priors, pi, tau, tab_jrx = main(self.graph_edges, n_clusters, max_iter = max_iter, method = initilisation_method)
        ICL_clusters = ICL(self.graph_edges, tau, pi, priors)
        result = {'pi': pi.to('cpu').numpy(), 'tau' : tau.to('cpu').numpy(), 'jrx' : tab_jrx, 'priors' : priors.to('cpu').numpy(), 'ICL' : ICL_clusters, 'max_iter' : max_iter, 'initialisation' : initilisation_method, 'n_clusters' : n_clusters}
        self.results[n_clusters] = result
        
    def fit(self, tab_n_clusters = [2,3,4,5,6,7,8], n_clusters = None, max_iter = None, initilisation_method = None, save_path = "save_results", print_fit_finish = True):
        """This function will fit the EM algorithm to your graph

        Args:
            tab_n_clusters (list, optional): A list of the number of clusters for which you want to perform your fit Defaults to [2,3,4,5,6,7,8].
            n_clusters (int, optional): The number of clusters for which you want to perform your fit
            max_iter (int, optional): The number of iterations for the EM algorithm. Defaults to 50.
            initilisation_method (str, optional): The initialisation method you want to use. Defaults to 'spectral'.
            save_path (str, optional): the folder in which you want to save the results. Defaults to "save_results".
            print_fit_finish (bool, optional): print "fit finished' after each fit for a certain number of clusters. Defaults to True.
        """
        if max_iter == None:
            max_iter = self.max_iter
        if initilisation_method == None:
            initilisation_method = self.initilisation_method
        if n_clusters == None:
            for n_cluster in tab_n_clusters:
                self.EM(n_cluster, max_iter, initilisation_method)
                if print_fit_finish:
                    print('Fit finished for ', n_cluster, ' clusters ')
        else :
            self.EM(n_clusters, max_iter, initilisation_method)
        with open(save_path+'.pkl', 'wb') as f:
            pickle.dump(self.results, f)
            
    def plot_jrx_several_plot(self, tab_n_clusters):
        """_summary_

        Args:
            tab_n_clusters (lis int): plot the J(R_X) functions for the input list of number of clusters
        """
        for n_clusters in tab_n_clusters:
            plot_JRX(self.results[n_clusters]['jrx'], n_clusters)
    
    def plot_jrx(self, save_path = None):
        """plot the J(R_X) functions for all the clusters used in the fit

        Args:
            save_path (srt, optional): the name of the file you want to save. Defaults to None.
        """
        for n_clusters in self.results.keys():
            # Tracer le graphique en fonction des indices
            plt.plot(self.results[n_clusters]['jrx'], label=f'{n_clusters} clusters') 

        plt.title(r'$\mathcal{J}(R_{\mathcal{X}})$ values')
        plt.xlabel('Iterations')
        plt.ylabel(r'$\mathcal{J}(R_{\mathcal{X}})$')

        plt.legend()

        if save_path != None:
            plt.savefig(f'{save_path}.png')
            plt.close()
        else : 
            plt.show()
            
    def plot_icl(self, save_path = None):
        """plot the ICL value (for one iteration) of all the n_clusters used in the fit

        Args:
            save_path (srt, optional): the name of the file you want to save. Defaults to None.
        """
        tab_clusters = []
        tab_ICL = []
        for n_clusters in self.results.keys():
            tab_clusters.append(n_clusters)
            tab_ICL.append(self.results[n_clusters]['ICL'])
        plot_ICL(tab_clusters, tab_ICL, save_path)
    
    def plot_adjency_matrix(self, n_clusters, save_path = None, show_names = False):
        """plot the reordered adjency matrix of the graph for the final results

        Args:
            n_clusters (int): plot the adjency matrix for the result of the EM with n_clusters classes
            save_path (srt, optional): the name of the file you want to save. Defaults to None.
            show_names (bool, optional): Plot the names of the nodes in the adjency matrix. Defaults to False.
        """
        # Get the node estimated distribition
        z = from_tau_to_Z(torch.from_numpy(self.results[n_clusters]['tau']))
        cluster_indices = {q: np.where(z[:, q] == 1)[0] for q in range(n_clusters)}
        
        # Permutation of the adjency matrix
        adjacency_matrix = nx.to_numpy_array(self.graph)
        new_order = np.concatenate([cluster_indices[q] for q in range(n_clusters)])
        permuted_matrix = adjacency_matrix[np.ix_(new_order, new_order)]
        
        plt.figure(figsize=(6, 6))
        plt.spy(permuted_matrix, markersize=0.5)
        
        if show_names:
            names_index = [(index, name) for index, name in enumerate(self.graph.nodes())]
            index_to_name = dict(names_index)
            labels = [index_to_name[index] for index in new_order]
            plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=6)  # Rotate for better legibility
            plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=6)

        # Add the delimitaitons between the clusters
        current_idx = 0
        for q in range(n_clusters):
            cluster_size = len(cluster_indices[q])
            if cluster_size > 0:
                current_idx += cluster_size
                plt.axvline(x=current_idx - 0.5, color='r', linestyle='--')
                plt.axhline(y=current_idx - 0.5, color='r', linestyle='--')

        plt.title("Adjency matrix with nodes grouped in clusters")
        if save_path != None:
            plt.savefig(f'{save_path}.png')
            plt.close()
        else : 
            plt.show()
            
    def plot_all_adjency_matrices(self, save_path = None, show_names = False):
        """plot all the reordered adjency matrix of the graph for the final results 

        Args:
            save_path (srt, optional): the name of the file you want to save. Defaults to None.
            show_names (bool, optional): Plot the names of the nodes in the adjency matrix. Defaults to False.
        """
        for n_clusters in self.results.keys():
            if save_path != None:
                self.plot_adjency_matrix(n_clusters, save_path + "_"+ str(n_clusters)+"_clusters", show_names = show_names)
            else:
                self.plot_adjency_matrix(n_clusters, save_path = None, show_names = show_names)
        
    def load_results(self, results_path):
        """Load the results from a pkl file of a previous model

        Args:
            results_path (str): where to find the pkl file
        """
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)

    def get_clusters(self, n_clusters):
        """Return a list with the nodes of each classes

        Args:
            n_clusters (int): the number of classes for which you want to see the results

        Returns:
            dic: a dictionnary with the nodes in each classes
        """
        z = from_tau_to_Z(torch.from_numpy(self.results[n_clusters]['tau']))
        nodes_index=[(index, node) for index, node in enumerate(self.graph.nodes())]
        cluster_indices = {q: np.where(z[:, q] == 1)[0] for q in range(n_clusters)}
        clusters={}
        for q, indices in cluster_indices.items():
            clusters[q] = [item[1] for item in nodes_index if item[0] in indices] 
        return clusters
    
    def precise_fit_one_method(self, list_clusters, nbr_iter_per_cluster = 20, max_iter_em = 50, initialisation_method = 'random'):
        ICL_values_method = {}
        for n_cluster in list_clusters:
                ICL_values_method[n_cluster] = []
        for _ in range(nbr_iter_per_cluster):
            self.fit(tab_n_clusters=list_clusters, max_iter=max_iter_em, initilisation_method=initialisation_method, print_fit_finish=False)
            for n_cluster in list_clusters:
                ICL_values_method[n_cluster].append(self.results[n_cluster]['ICL'])
        self.ICL_values[initialisation_method] = ICL_values_method
        self.parameters = {}
        self.parameters['nbr_iter_per_cluster'] = nbr_iter_per_cluster
        self.parameters['max_iter_em'] = max_iter_em
        
    def precise_fit(self, list_clusters, nbr_iter_per_cluster = 20, max_iter_em = 50, list_initialisation_methods = ['random']):
        """Fit the model several times on several initialisation methods

        Args:
            list_clusters (list): the list of clusters you want to use for your fit
            nbr_iter_per_cluster (int, optional): the number of time you perform the EM per cluster. Defaults to 20.
            max_iter_em (int, optional): the max_iter for each EM algorithm. Defaults to 50.
            list_initialisation_methods (list, optional): the list of initialisation method you want to use. Defaults to ['random'].
        """
        for method in list_initialisation_methods:
            self.precise_fit_one_method(list_clusters, nbr_iter_per_cluster = nbr_iter_per_cluster, max_iter_em = max_iter_em, initialisation_method=method)
                
    def plot_repeated_ICL(self, list_methods = None, save_path = None):
        """Plot the ICL after a precise fit, with the confidence value of each result

        Args:
            list_methods (list, optional): the list of methods for which you want to plot the results. Defaults to None.
            save_path (str, optional): The file to save the results. Defaults to None.
        """
        # Calcul des moyennes, écarts-types, et intervalles de confiance
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        if list_methods ==  None:
            list_methods = list(self.ICL_values.keys())
        for idx, method in enumerate(list_methods):
            keys = list(self.ICL_values[method].keys())
            means = [np.mean(self.ICL_values[method][key]) for key in keys]
            std_devs = [np.std(self.ICL_values[method][key]) for key in keys]

            # Calcul de l'intervalle de confiance à 95%
            confidence_interval = [1.96 * std / np.sqrt(len(self.ICL_values[method][key])) for std, key in zip(std_devs, keys)]
            lower_bound = [mean - ci for mean, ci in zip(means, confidence_interval)]
            upper_bound = [mean + ci for mean, ci in zip(means, confidence_interval)]

            # Création du graphique
            plt.plot(keys, means, label=f"Method is {method}", color=colors[idx], marker='o')
            plt.fill_between(keys, lower_bound, upper_bound, color=colors[idx], alpha=0.2)
        
        plt.title(f"Average ICL values after {self.parameters['nbr_iter_per_cluster']} EM algorithms")
        plt.xlabel("Number of classes")
        plt.ylabel("ICL values")
        plt.legend()
        plt.grid(True)
        if save_path != None:
            plt.savefig(f'{save_path}.png')
            plt.close()
        else : 
            plt.show()
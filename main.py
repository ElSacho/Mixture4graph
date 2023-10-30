import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 
import random
import warnings
from scipy.sparse import csr_matrix
from tqdm import tqdm

class GraphGenerator():
    def __init__(self):
        pass
    
    def generate(self, size, pi, priors):
        """_summary_

        Args:
            size (int): number of nodes in the graph 
            pi (numpy.array): (n_cluster x n_cluster) matrix defined at the ebd of page 5
            priors (np.array): (n_cluster) priors distributions st : sum(priors) = 1 and sum(int(size * priors)) = size 
            
        Returns:
            graph generated with the Erdos-Renyi mixture or graphs model with size nodes 
        """
        # Create the graph and add the nodes
        G = nx.Graph()
        G.add_nodes_from(range(size))
        
        n_cluster = len(priors)
        
        # Get the number of nodes per cluster
        scaled_priors = priors * size
        number_of_node_per_cluster = np.round(scaled_priors).astype(int)
        cumulated_number_of_node_per_cluster = np.cumsum(number_of_node_per_cluster)
        
        # We try if the priors distributions are good 
        sum_rounded_priors = np.sum(number_of_node_per_cluster)
        sum_rounded_priors == size
        if sum_rounded_priors != size:
            print("Number of node(s) per cluster : ",number_of_node_per_cluster, " for a graph for size : ",size)
            
            raise Exception("The priors are not correctly scaled with the size")
    
        for cluster_i in range(n_cluster):
            if number_of_node_per_cluster[cluster_i] != 0:
                for cluster_j in range(cluster_i): # we need to treat the case where cluster_i = cluster_j after
                    if number_of_node_per_cluster[cluster_j] != 0:  
                        p_ij = pi[cluster_i, cluster_j]
                        if cluster_i == 0 : i_min = 0
                        else : i_min = cumulated_number_of_node_per_cluster[cluster_i - 1] 
                        i_max = i_min + number_of_node_per_cluster[cluster_i]
                        if cluster_j == 0 : j_min = 0
                        else : j_min = cumulated_number_of_node_per_cluster[cluster_j - 1] 
                        j_max = j_min + number_of_node_per_cluster[cluster_j]
                        connections_between_those_clusters = p_ij * np.ones((i_max - i_min, j_max - j_min))
                        connections_between_those_clusters = (np.random.rand(i_max - i_min, j_max - j_min) < connections_between_those_clusters).astype(int)
                        shape_i, shape_j = connections_between_those_clusters.shape
                        for node_i in range(shape_i):
                            for node_j in range(shape_j):
                                if connections_between_those_clusters[node_i, node_j] == 1:
                                    G.add_edge(node_i + i_min, node_j + j_min)
        
        for cluster_i in range(n_cluster):
            if number_of_node_per_cluster[cluster_i] != 0:           
                p_ii = pi[cluster_i, cluster_i]
                if cluster_i == 0 : i_min = 0
                else : i_min = cumulated_number_of_node_per_cluster[cluster_i - 1] 
                i_max = i_min + number_of_node_per_cluster[cluster_i]
                connections_between_those_clusters = p_ii * np.ones((i_max - i_min, i_max - i_min))
                connections_between_those_clusters = (np.random.rand(i_max - i_min, i_max - i_min) < connections_between_those_clusters).astype(int)
                for node_i in range(i_max - i_min):
                    for node_j in range(node_i): #we juste need to add the values for j < i as the graph is undirected and with no self loop
                        if connections_between_those_clusters[node_i, node_j] == 1:
                            G.add_edge(node_i + i_min, node_j + i_min)
                            
        return G
    
                    
    def legacy_generate(self, size, pi, priors):
        """_summary_

        Args:
            size (int): number of nodes in the graph 
            pi (numpy.array): (n_cluster x n_cluster) matrix defined at the ebd of page 5
            priors (np.array): (n_cluster) priors distributions st : sum(priors) = 1 and sum(int(size * priors)) = size 
            
        Returns:
            graph generated with the Erdos-Renyi mixture or graphs model with size nodes 
        """
        # Create the graph and add the nodes
        G = nx.Graph()
        G.add_nodes_from(range(1, size+1))
        
        n_cluster = len(priors)
        
        # Get the number of nodes per cluster
        scaled_priors = priors * size
        number_of_node_per_cluster = np.round(scaled_priors).astype(int)
        cumulated_number_of_node_per_cluster = np.cumsum(number_of_node_per_cluster)
        
        # We try if the priors distributions are good 
        sum_rounded_priors = np.sum(number_of_node_per_cluster)
        sum_rounded_priors == size
        if sum_rounded_priors != size:
            raise Exception("The priors are not correctly scaled with the size")
        
        # Idee 2 : on fait une matrice de taille sizexsize avec les probas partout
        proba_edges_connecting = np.zeros((size, size))
        for cluster_i in range(n_cluster):
            if number_of_node_per_cluster[cluster_i] != 0:
                for cluster_j in range(cluster_i + 1):
                    if number_of_node_per_cluster[cluster_j] != 0:  
                        p_ij = pi[cluster_i, cluster_j]
                        i_min = cumulated_number_of_node_per_cluster[cluster_i]
                        j_min = cumulated_number_of_node_per_cluster[cluster_j]
                        if cluster_i < n_cluster - 1: i_max = cumulated_number_of_node_per_cluster[cluster_i + 1] - 1
                        else : i_max = size - 1
                        if cluster_j < n_cluster - 1: j_max = cumulated_number_of_node_per_cluster[cluster_j + 1] - 1
                        else : j_max = size - 1
                        proba_edges_connecting[i_min:i_max,j_min:j_max] = p_ij # just one side but then the matrix is gonna be symetric
                        # proba_edges_connecting[j_min:j_max,i_min:i_max] = p_ij
                        # Question pour le rapport : est-ce interessant de calculer l'esperance du determinant ? 
        edges = (np.random.rand(size, size) < proba_edges_connecting).astype(int)
        for i in range(size):
            for j in range(i+1):
                if edges[i, j] == 1:
                    G.add_edge(i, j)
        
        return G
        
        
    def find_the_parameters(self, graph, solver):
        pass
    
class Solver():
    def __init__(self):
        pass
    
    def get_theta_from_tau(self, tau):
        n_nodes, _ = tau.shape
        tau_replicated = np.repeat(tau[:, np.newaxis, :, np.newaxis], n_nodes, axis=1)
        theta = tau_replicated * tau_replicated.transpose((1, 0, 3, 2))
                
        return theta # I verified this function
    
    def E(self, tau, graph_edges, priors, pi, eps=1e-6, max_iter = 10):
        """_summary_

        Args:
            graph (_type_): _description_
            pi (_type_): _description_
            priors (_type_): _description_

        Returns:
            tau - np.array: approximation of the priors, size n_clusters
        """
        new_tau = self.fixed_point_function(tau, graph_edges, priors, pi)        
        difference_matrix = np.abs(new_tau - tau)
        convergence_finished = np.all(difference_matrix < eps)
        tau = new_tau.copy()
        
        for i in tqdm(range(max_iter)):
            new_tau = self.fixed_point_function(tau, graph_edges, priors, pi)
            difference_matrix = np.abs(new_tau - tau)
            convergence_finished = np.all(difference_matrix < eps)
            tau = new_tau.copy()
            if convergence_finished:
                break    
        print(np.mean(difference_matrix))
        theta = self.get_theta_from_tau(tau)
        return tau, theta
    
    def is_tau_convergence_ok(self, tau, new_tau, eps):
        difference_matrix = np.abs(new_tau - tau)
        return np.all(difference_matrix < eps)
        
    def fixed_point_function2(self, tau, graph_edges, priors, pi ):
        """_summary_

        Args:
            tau (np.array): (n_nodes, n_clusters) last estimation au tau
            graph_edges (np.array): (n_nodes, n_nodes) graph edges matrix
            priors (np.array): (n_clusters) P(Z_ik = 1) for k
            pi (np.array): (n_clusters, n_clusters) 

        Returns:
            _type_: _description_
        """
        
        new_tau = np.zeros_like(tau)
        n_nodes, n_nodes = graph_edges.shape
        n_clusters = len(priors)
        
        # exp_term = (pi ** graph_edges[:, :,np.newaxis, np.newaxis]) * ((1 - pi) **(1- graph_edges[:, :,np.newaxis, np.newaxis])) # This one works I have verifide
        # M = exp_term[:, :, :, :] ** tau[:, np.newaxis, np.newaxis, :]
                
        for i in range(n_nodes):
            for l in range(n_clusters):
                p = 1
                for j in range(n_nodes):
                    if j!=i:
                        for k in range(n_clusters):
                            val = (pi[k,l]**graph_edges[i,j]) * (1 - pi[k,l])**(1-graph_edges[i,j]) 
                            p *= val ** tau[j,l]
                # print(p)
                new_tau[i,l] = priors[l] * p
                
        new_tau = self.normalize_tau(new_tau)
        # product_axis3 = np.prod(M, axis=3)
        
        return new_tau  
    
    def fixed_point_function(self, tau, graph_edges, priors, pi ):
        """_summary_

        Args:
            tau (np.array): (n_nodes, n_clusters) last estimation au tau
            graph_edges (np.array): (n_nodes, n_nodes) graph edges matrix
            priors (np.array): (n_clusters) P(Z_ik = 1) for k
            pi (np.array): (n_clusters, n_clusters) 

        Returns:
            _type_: _description_
        """
        
        new_tau = np.zeros_like(tau)
        
        exp_term = (pi ** graph_edges[:, :,np.newaxis, np.newaxis]) * ((1 - pi) **(1- graph_edges[:, :,np.newaxis, np.newaxis])) # This one works I have verifide
        K = exp_term ** tau[np.newaxis, :, np.newaxis, :] # K[i,j,k,l] = M[i,j,k,l] ** tau[j,l]
        K = np.prod(K, axis=3)
        
        for i in range(K.shape[0]):
            K[i, i, :] = 1

        new_tau = np.prod(K, axis=1)
        new_tau = new_tau * priors
        new_tau = self.normalize_tau(new_tau)
        return new_tau
   
    def M2(self, graph_edges, tau, theta):
        """_summary_

        Args:
            graph_edges (np.array): the graph we are studiying shape (n_nodes, n_nodes)
            tau (np.array): tau[i,q] = P(Z_iq = 1) i in node, q in cluster shape of (n_nodes, n_clusters)
            theta (np.array): shape of (n_nodes, n_nodes, n_clusters, n_clusters)

        Returns:
            priors np.array: (n_clusters)
            pi np.array: (n_clusters, n_clusters)
        """
        # Get the prior
        priors = np.mean(tau, axis=0) # (n_clusters)
        
        n_clusters = len(priors)
        n_nodes, _ = graph_edges.shape
        
        thetaX = np.zeros((n_nodes, n_nodes, n_clusters, n_clusters))
        divided = np.zeros((n_nodes, n_nodes, n_clusters, n_clusters))
        for l in range(n_clusters):
            for q in range(n_clusters):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        thetaX[i,j,q,l] = tau[i,q]*tau[j,l]*graph_edges[i,j]
                        divided[i,j,q,l] = tau[i,q]*tau[j,l]
            
        thetaX = np.sum(np.sum(thetaX, axis=1), axis=0) # (n_clusters, n_clusters)
        divided = np.sum(np.sum(thetaX, axis=1), axis=0) 
        # Get the denominator of equation in 5.3
        # divided = np.sum(np.sum(theta, axis=1), axis=0) # (n_clusters, n_clusters)
        # Get the approximation for pi              
        pi = thetaX / divided
        
        # thetaX2 = theta * graph_edges[:, :, np.newaxis, np.newaxis] # (n_nodes, n_nodes, n_clusters, n_clusters)
        # thetaX2 = np.sum(np.sum(thetaX2, axis=1), axis=0) # (n_clusters, n_clusters)
        # # Get the denominator of equation in 5.3
        # divided2 = np.sum(np.sum(theta, axis=1), axis=0) # (n_clusters, n_clusters)
        # pi2 = thetaX2 / divided2
        # print("The two methods in the M algorithm to calculate pi are egale ? : ", pi == pi2)

        return priors, pi
  
    def M(self, graph_edges, tau, theta):
        """_summary_

        Args:
            graph_edges (np.array): the graph we are studiying shape (n_nodes, n_nodes)
            tau (np.array): tau[i,q] = P(Z_iq = 1) i in node, q in cluster shape of (n_nodes, n_clusters)
            theta (np.array): shape of (n_nodes, n_nodes, n_clusters, n_clusters)

        Returns:
            priors np.array: (n_clusters)
            pi np.array: (n_clusters, n_clusters)
        """
        # Get the prior
        priors = np.mean(tau, axis=0) # (n_clusters)
        
        # Get the nominator of equation in 5.3
        thetaX = theta * graph_edges[:, :, np.newaxis, np.newaxis] # (n_nodes, n_nodes, n_clusters, n_clusters)    
        thetaX = np.sum(np.sum(thetaX, axis=1), axis=0) # (n_clusters, n_clusters)
        # Get the denominator of equation in 5.3
        divided = np.sum(np.sum(thetaX, axis=1), axis=0) # (n_clusters, n_clusters)
        
        # Get the approximation for pi              
        pi = thetaX / divided # (n_clusters, n_clusters)
        return priors, pi
         
    def normalize_tau(self, tau):
        tau = tau / tau.sum(axis=1, keepdims=True)
        return tau  # je suis sur de ça
        
    def EM_algorithm(self, graph, n_clusters, n_iter = 500):
        n_nodes = len(graph.nodes)
        graph_edges = np.zeros((n_nodes, n_nodes), dtype=int)
        for i, j in graph.edges:
            graph_edges[i, j] = 1
            graph_edges[j, i] = 1 
            
        tau = np.random.uniform(0, 1, size=(n_nodes, n_clusters))
        tau = np.ones((n_nodes, n_clusters))
        tau = self.normalize_tau(tau)
        theta = self.get_theta_from_tau(tau)
                
        for i in range(n_iter):
            # M step
            priors, pi = self.M(graph_edges, tau, theta)
            # print(pi.shape)
            # print(priors.shape)
            # E step
            tau, theta = self.E(tau, graph_edges, priors, pi)
            # print(tau)
            # print(theta.shape)
            
        return priors, pi

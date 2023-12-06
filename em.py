import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt




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
    
    pi = np.zeros((n_cluster, n_cluster))
    for q in range(n_cluster - 1):
        for l in range(q+1, n_cluster):
            denominator = 0
            nominator = 0
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if j != i:
                       nominator += tau[i,q] * tau[j,l] * X[i,j]
                       denominator += tau[i,q] * tau[j,l]
            pi[q,l] = nominator/denominator
            pi[l,q] = nominator/denominator
    
    return prior, pi


def b(X_ij, pi_ql):
    """
    Let us discuss when b=0
    if b = 0 
        pi_ql = 0
            then one node between the cluster q and the cluster l cannot be connected
            then if i and j simultaneously have nothing to do with this cluster : there is no link and it can be equal to zero
            if i is in cluster q and j in cluster l then X_ij = 0 so pi_ql**X_ij = 0
        if pi_ql = 1
            then one node between the cluster q and the cluster l have to be connected
            then if i and j simultaneously have nothing to do with this cluster : there is no link and it can be equal to zero
            if i is in cluster q and j in cluster l then X_ij = 1 so (1 - pi_ql)**(1-X_ij) = 0
        
    Args:
        X_ij (int): 1 if there is an edge between i and j; 0 otherwise
        pi_ql (_type_): probability that one node in cluster q is connected to one node of cluster l

    Returns:
        b : the likehood of the observation
    """
    
    b = pi_ql**X_ij * (1-pi_ql)**(1-X_ij)
    return b

def fixed_function()
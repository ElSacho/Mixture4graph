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
    
    clustering = {}
    
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
    
    pi = np.zeros((n_cluster, n_cluster))
    for q in range(n_cluster):
        for l in range(n_cluster):
            denominator = 0
            nominator = 0
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if j != i:
                       nominator += tau[i,q] * tau[j,l] * X[i,j]
                       denominator += tau[i,q] * tau[j,l]
            pi[q,l] = nominator/denominator
            pi[l,q] = nominator/denominator
    
    # pi = np.zeros((n_cluster, n_cluster))
    # for q in range(n_cluster):
    #     for l in range(q, n_cluster):
    #         denominator = 0
    #         nominator = 0
    #         for i in range(n_nodes):
    #             for j in range(n_nodes):
    #                 if j != i:
    #                    nominator += tau[i,q] * tau[j,l] * X[i,j]
    #                    denominator += tau[i,q] * tau[j,l]
    #         pi[q,l] = nominator/denominator
    #         pi[l,q] = nominator/denominator
    return prior, pi


def b_pow(X_ij, pi_ql, tau_jl):
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
    
    b = (pi_ql**X_ij)**tau_jl * ((1-pi_ql)**(1-X_ij)) ** tau_jl
    return b


def fixed_function_i_q(i, q, old_tau, X, pi, priors, norm):
    n_nodes, n_clusters = old_tau.shape
    new_tau_i = np.zeros(n_clusters)
    
    s = 0
    val_iq = 1
    for j in range(n_nodes):
        for l in range(n_clusters):
            if j != i :
                b = b_pow(X[i,j], pi[q,l], old_tau[j,l])
                if b != 0:
                    val_iq *= b
                else : val_iq *= b

    return priors[q] * val_iq / norm

def appro_all_in_one(tau, X, pi, priors, eps = 1e-04, max_iter = 50):
    n_nodes = X.shape[0]
    n_clusters = pi.shape[0]
    
    finish = False
    current_iter = 0

    while not finish and current_iter < max_iter:
        
        old_tau = tau.copy()
        for i in range(n_nodes):
            norm_i = tau.sum(axis=1, keepdims=True)[i]
            for q in range(n_clusters):
                for j in range(n_nodes):
                    for l in range(n_clusters):
                        if j != i :
                            b = b_pow(X[i,j], pi[q,l], old_tau[j,l])
                            if b != 0:
                                val_iq *= b
                            else : val_iq *= b
                
                
                
                tau[i,q] = fixed_function_i_q(i, q, old_tau, X, pi, priors, 1)
        
        tau = tau / tau.sum(axis=1, keepdims=True)
        # print(tau)
        difference_matrix = np.abs(tau - old_tau)
        if np.all(difference_matrix < eps):
            finish = True    
        current_iter += 1

    return tau

def approximate_tau_step_by_step(tau, X, pi, priors, eps = 1e-04, max_iter = 50):
    
    n_nodes = X.shape[0]
    n_clusters = pi.shape[0]
    
    finish = False
    current_iter = 0

    while not finish and current_iter < max_iter:
        
        old_tau = tau.copy()
        for i in range(n_nodes):
            norm_i = tau.sum(axis=1, keepdims=True)[i]
            for q in range(n_clusters):
                tau[i,q] = fixed_function_i_q(i, q, old_tau, X, pi, priors, 1)
        
        tau = tau / tau.sum(axis=1, keepdims=True)
        # print(tau)
        difference_matrix = np.abs(tau - old_tau)
        if np.all(difference_matrix < eps):
            finish = True    
        current_iter += 1

    return tau

def fixed_function(old_tau, X, pi, priors):
    n_nodes, n_clusters = old_tau.shape
    new_tau = np.zeros_like(old_tau)
    
    for i in range(n_nodes):
        for q in range(n_clusters):
            val_iq = 1
            for j in range(n_nodes):
                for l in range(n_clusters):
                    if j != i :
                        b = b_pow(X[i,j], pi[q,l], old_tau[j,l])
                        if b == 0:
                            continue
                            # print("X : ", X[i,j], 'for (i,j) = ', i,j)
                            # print("pi[q,l] : ", pi[q,l], 'for (q,l) = ', q,l)
                            # print("tau ; ", old_tau[j,l])
                            # or we store those indexes and then we will change the values
                            # new_tau[i,q] = 0
                            # new_tau[i,l] = 0
                            # new_tau[j,q] = 0
                            # new_tau[j,l] = 0
                        else:
                            val_iq *= b
            new_tau[i, q] = priors[q] * val_iq
            # print(new_tau[i,q])
    
    return new_tau
    
def approximate_tau(old_tau, X, pi, priors, eps = 1e-04, max_iter = 50):
    
    old_tau = np.random.uniform(0, 1, size=(X.shape[0], pi.shape[0]))
    old_tau = old_tau / old_tau.sum(axis=1, keepdims=True)
    current_iter = 0
    
    while current_iter < max_iter:
        # print("old tau : ")
        # print(old_tau)
        new_tau = fixed_function(old_tau, X, pi, priors)
        # print("new_tau after funtion :")
        # print(new_tau)
        # print(" ")
        # new_tau = new_tau / new_tau.sum(axis=1, keepdims=True)
        
        difference_matrix = np.abs(new_tau - old_tau)
        if np.all(difference_matrix < eps):
            break
        # print("In the fixed function, the appriximation for tau is at : ", np.mean(difference_matrix))
        old_tau = new_tau.copy()
        
        current_iter += 1
    print("")
    # print("tau :")
    # print(new_tau)
    print("n_iter : ", current_iter)
    return new_tau
    
def main(X, n_clusters, max_iter):
    n_nodes, _ = X.shape
    # pi = np.random.uniform(0, 1, size=(n_clusters, n_clusters))
    # pi = (pi + pi.T) / 2
    G = nx.from_numpy_array(X)
    tau = spectral_clustering(G, n_clusters)
    print(np.mean(tau, axis = 0))
    # priors = np.random.uniform(0, 1, size=n_clusters)
    # priors = priors / np.sum(priors)
    
    # tau = np.random.uniform(0, 1, size=(n_nodes, n_clusters))
    # tau = tau / tau.sum(axis=1, keepdims=True)
    # print("tau")
    # print(tau)
    
    current_iter = 0
    while current_iter < max_iter:
        priors, pi = return_priors_pi(X, tau.copy())
        # print(priors)
        tau = approximate_tau_step_by_step(tau.copy(), X, pi.copy(), priors.copy())
        # print(tau)
        # print("pi : \n",pi)
        # tau = approximate_tau(tau, X, pi, priors)
        # tau = tau / tau.sum(axis=1, keepdims=True)
        # print("tau")
        # print(tau)
        current_iter += 1
    return priors, pi

def main_with_tau(X, tau, n_clusters, max_iter):
    current_iter = 0
    while current_iter < max_iter:
        priors, pi = return_priors_pi(X, tau)
        # print(priors)
        tau = approximate_tau_step_by_step(tau, X, pi, priors)
        # print(tau)
        # print("pi : \n",pi)
        # tau = approximate_tau(tau, X, pi, priors)
        # tau = tau / tau.sum(axis=1, keepdims=True)
        # print("tau")
        # print(tau)
        current_iter += 1
    return priors, pi

def get_X_from_graph(graph):
    n_nodes = len(graph.nodes)
    graph_edges = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in graph.edges:
        graph_edges[i, j] = 1
        graph_edges[j, i] = 1
        
    return graph_edges
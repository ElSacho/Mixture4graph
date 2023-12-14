import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import random

def generate(n_vertices, pi, priors):
    G=nx.Graph()
    G.add_nodes_from(range(n_vertices))
    
    #attribute a cluster to every nodes, with proba alpha_p
    vertices_clusters=[np.random.choice( np.arange( 0,len(priors) ), p = priors ) for _ in range(n_vertices)]
    
    for i in range(n_vertices):
        for j in range(i+1,n_vertices):
            proba_link_ij = pi[vertices_clusters[i]][vertices_clusters[j]]
            if random.random() < proba_link_ij:
                G.add_edge(i,j)
    
    return G

def generate_and_give_tau(n_vertices, pi, priors):
    G=nx.Graph()
    G.add_nodes_from(range(n_vertices))
    
    #attribute a cluster to every nodes, with proba alpha_p
    vertices_clusters=[np.random.choice( np.arange( 0,len(priors) ), p = priors ) for _ in range(n_vertices)]
    
    for i in range(n_vertices):
        for j in range(i+1,n_vertices):
            proba_link_ij = pi[vertices_clusters[i]][vertices_clusters[j]]
            if random.random() < proba_link_ij:
                G.add_edge(i,j)
    tau = format_tau(vertices_clusters, len(priors))
    return G, tau

def format_tau(vertices_clusters, n_clusters):
    tau = np.zeros((len(vertices_clusters), n_clusters))
    for i in range(len(vertices_clusters)):
        tau[i,vertices_clusters[i]] = 1
    return tau

        
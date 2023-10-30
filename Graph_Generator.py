import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
import random


class GraphGenerator(nx.Graph):
    def __init__(self,n_clusters,pi,alphas):
        super().__init__()
        self.n_clusters=n_clusters #number of different clusters in the graph
        self.pi=pi #matrix of connexions probabilities. pi is symmetric. pi is n_clusters x n_clusters big.
        self.alphas=alphas # alphas is a vector of size n_clusters
    
    def generate(self,n_vertices): #return a graph with n_vertices vertices and whose edges are characterized by pi
        G=nx.Graph()
        G.add_nodes_from([range(n_vertices)])
        #attribute a cluster to every nodes, with proba alpha_p
        vertices_clusters=[np.random.choice(np.arange(0,self.n_clusters),p=self.alphas) for _ in range(n_vertices)]
        for i in range(n_vertices):
            for j in range(i+1,n_vertices):
                proba=self.pi[vertices_clusters[i]][vertices_clusters[j]]
                if random.random()<proba:
                    G.add_edge(i,j)
        return G
    

    def show_graph(self, n_vertices=10,node_size = 300, font_size=10):
        # Utilisez l'algorithme de disposition du ressort pour positionner les nœuds
        G=self.generate(n_vertices)

        pos = nx.spring_layout(G)

        # Dessinez le graphe en utilisant les positions calculées
        nx.draw(G, pos, node_size=node_size, node_color='skyblue', font_size=font_size)

        # Affichez le graphe
        plt.show()




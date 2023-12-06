import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
# import random


def show_graph(G, with_labels=True, node_size = 300, font_size=10):
    # Utilisez l'algorithme de disposition du ressort pour positionner les nœuds
    pos = nx.spring_layout(G)

    # Dessinez le graphe en utilisant les positions calculées
    nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color='skyblue', font_size=font_size)

    # Affichez le graphe
    plt.show()
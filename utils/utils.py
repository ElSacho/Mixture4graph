import networkx as nx
import matplotlib.pyplot as plt
import random


def show_graph(G, with_labels=True, node_size = 300, font_size=10):
    # Utilisez l'algorithme de disposition du ressort pour positionner les nœuds
    pos = nx.spring_layout(G)

    # Dessinez le graphe en utilisant les positions calculées
    nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color='skyblue', font_size=font_size)

    # Affichez le graphe
    plt.show()


def generate_random_graph(size):
    # Créez un graphe vide
    G = nx.Graph()

    # Ajoutez n nœuds au graphe
    G.add_nodes_from(range(1, size+1))

    # Ajoutez des arêtes de manière aléatoire
    for node1 in G.nodes:
        for node2 in G.nodes:
            # Assurez-vous que vous n'ajoutez pas une arête vers le même nœud
            if node1 != node2:
                # Vous pouvez utiliser une condition aléatoire pour décider d'ajouter une arête
                if random.random() < 0.3:  # Probabilité d'ajouter une arête (ici, 30%)
                    G.add_edge(node1, node2)
    return G
import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
# import random

def plot_JRX(jrx, n_clusters = None):
    # Créer une liste d'indices pour l'axe des x
    # indices = list(range(len(jrx)))
    if n_clusters == None:
        title = r'The $\mathcal{J}(R_{\mathcal{X}})$  values'
    else :
        title = r'The $\mathcal{J}(R_{\mathcal{X}})$ values for ' + str(n_clusters) + ' clusters'
        
    # Tracer le graphique en fonction des indices
    plt.plot(jrx)
    plt.title(title)
    # Ajouter des étiquettes aux axes
    plt.xlabel('Iterations')
    plt.ylabel(r'$\mathcal{J}(R_{\mathcal{X}})$')

    # Afficher le graphique
    plt.show()
    
def plot_ICL(tab_n_clusters, tab_ICL):
    # Créer le graphique
    plt.plot(tab_n_clusters, tab_ICL, marker='o', linestyle='-')
    plt.xlabel('Number of clusters')
    plt.ylabel('ICL Value')
    plt.title('ICL with respect to the number of clusters')
    plt.grid(True)

    # Afficher le graphique
    plt.show()

def show_graph(G, with_labels=True, node_size = 300, font_size=10):
    # Utilisez l'algorithme de disposition du ressort pour positionner les nœuds
    pos = nx.spring_layout(G)

    # Dessinez le graphe en utilisant les positions calculées
    nx.draw(G, pos, with_labels=with_labels, node_size=node_size, node_color='skyblue', font_size=font_size, width = 0.1)

    # Affichez le graphe
    plt.show()
    
def show_multiple_graph(list_G, list_titles=None, with_labels=True, node_size=300, font_size=10):
    n = len(list_G)  # Nombre de graphes dans la liste
    if n == 0:
        return  # Pas de graphes à afficher
    if list_titles == None:
        list_titles = [f"The graph number {i}" for i in range(len(list_G))] 
    # Assurez-vous que la liste des titres a la même longueur que la liste des graphes
    if len(list_titles) != n:
        raise ValueError("La longueur de list_titles doit être égale à celle de list_G")
    
    # Créez une grille de sous-figures alignées horizontalement
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(n*4, 4))

    # Pour chaque graphe dans la liste
    for k, G in enumerate(list_G):
        ax = axes[k] if n > 1 else axes  # Gérer le cas où n == 1
        pos = nx.spring_layout(G)  # Position des nœuds

        # Dessinez le graphe dans la sous-figure correspondante
        nx.draw(G, pos, ax=ax, with_labels=with_labels, node_size=node_size, node_color='skyblue', font_size=font_size, width = 0.1)

        # Ajoutez le titre à la sous-figure
        ax.set_title(list_titles[k])

    plt.show()
    
def show_graph_cluster_color(graph, tau): # Déterminer des couleurs uniques pour chaque cluster
    clusters = np.argmax(tau, axis = 1)
    unique_clusters = set(clusters)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    cluster_color = {cluster: color for cluster, color in zip(unique_clusters, colors)}

    # Assigner une couleur à chaque nœud
    node_colors = [cluster_color[clusters[i]] for i in range(graph.number_of_nodes())]

    # Dessiner le graphe
    nx.draw(graph, node_color=node_colors, with_labels=True, node_size=30, width = 0.1)
    plt.show()

def from_tau_to_Z(tau):
    max_values = np.max(tau, axis=1, keepdims=True)
    mask = (tau == max_values)
    z = np.zeros_like(tau)
    z[mask] = 1
    return z

def plot_adjency_matrix(graph, tau, n_clusters):
    # Nous allons créer une matrice d'adjacence d'exemple avec des blocs pour simuler les clusters
    z = from_tau_to_Z(tau)
    cluster_indices = {q: np.where(z[:, q] == 1)[0] for q in range(n_clusters)}
    names_index=[(index, name) for index, name in enumerate(graph.nodes())]

    # Permuter la matrice d'adjacence
    adjacency_matrix = nx.to_numpy_array(graph)
    new_order = np.concatenate([cluster_indices[q] for q in range(n_clusters)])
    index_to_name = dict(names_index)

    # Reorder the list of the names of the nodes based on new_order
    reordered_list = [(index, index_to_name[index]) for index in new_order]
    permuted_matrix = adjacency_matrix[np.ix_(new_order, new_order)]
    
    # Visualisation de la matrice d'adjacence triée
    plt.figure(figsize=(6, 6))
    plt.spy(permuted_matrix, markersize=0.5)

    labels = [index_to_name[index] for index in new_order]
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=90, fontsize=6)  # Rotate for better legibility
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=6)


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

def get_clusters(G, tau, n_clusters):
    z = from_tau_to_Z(tau)
    nodes_index=[(index, node) for index, node in enumerate(G.nodes())]
    cluster_indices = {q: np.where(z[:, q] == 1)[0] for q in range(n_clusters)}
    clusters={}
    for q, indices in cluster_indices.items():
        clusters[q] = [item[1] for item in nodes_index if item[0] in indices] 
    return clusters
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
    
def plot_ICL(tab_n_clusters, tab_ICL, save_path = None):
    # Créer le graphique
    plt.plot(tab_n_clusters, tab_ICL, marker='o', linestyle='-')
    plt.xlabel('Number of clusters')
    plt.ylabel('ICL Value')
    plt.title('ICL with respect to the number of clusters')
    plt.grid(True)

    # Afficher le graphique
    if save_path != None:
        plt.savefig(f'{save_path}.png')
        plt.close()
    else :
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
    
def show_colors_multiple_graph(list_G, list_tau, list_titles=None, with_labels=True, node_size=300, font_size=10):
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
        
        clusters = np.argmax(list_tau[k], axis = 1)
        unique_clusters = set(clusters)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        cluster_color = {cluster: color for cluster, color in zip(unique_clusters, colors)}

        # Assigner une couleur à chaque nœud
        node_colors = [cluster_color[clusters[i]] for i in G.nodes()]

        # Dessiner le graphe
        nx.draw(G, pos, ax=ax, node_color=node_colors, with_labels=False, node_size=30, width = 0.1)
       
        
        

        # Dessinez le graphe dans la sous-figure correspondante
        # nx.draw(G, pos, ax=ax, with_labels=with_labels, node_size=node_size, node_color='skyblue', font_size=font_size, width = 0.1)

        # Ajoutez le titre à la sous-figure
        ax.set_title(list_titles[k])

    plt.show()
    
def show_graph_cluster_color(graph, tau): # Déterminer des couleurs uniques pour chaque cluster
    clusters = np.argmax(tau, axis = 1)
    unique_clusters = set(clusters)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    cluster_color = {cluster: color for cluster, color in zip(unique_clusters, colors)}

    # Assigner une couleur à chaque nœud
    node_colors = [cluster_color[clusters[i]] for i in graph.nodes()]

    # Dessiner le graphe
    nx.draw(graph, node_color=node_colors, with_labels=False, node_size=30, width = 0.1)
    plt.show()
    

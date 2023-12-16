import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def get_connected_component(graph):
    return graph.subgraph(max(nx.connected_components(graph), key=len))
    
def print_generalitize(graph):
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    print('The graph has ',n_nodes, 'nodes, and ', n_edges, ' edges')  
    print("The graph has ", nx.number_connected_components(graph)," connected components.")
    largest_cc = max(nx.connected_components(graph), key=len)

    print("The largest connected component in the graph has ", len(largest_cc), 'nodes')

    subG = graph.subgraph(largest_cc)
    print("The largest connected component in the graph has ", subG.number_of_edges(), ' edges')
    
    # Degree
    degree_sequence = [graph.degree(node) for node in graph.nodes()]

    min_deg = np.min(degree_sequence)
    max_deg = np.max(degree_sequence)
    mean_deg = np.mean(degree_sequence)
    med_deg = np.median(degree_sequence)

    print("The minimum degree of a node is : ", min_deg )
    print("The maximum degree of a node is : ", max_deg )
    print("The mean degree of a node is : ", mean_deg)
    print("The median degree of a node is : ", med_deg )
    
    clustering_coef = nx.transitivity(graph)
    print("The clustering coefficient of the graph G is : ", clustering_coef)
    
def plot_figure_graph(graph, save_path = None):
    fig = plt.figure("Degree of the graph", figsize=(8, 4))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)

    ax1 = fig.add_subplot(axgrid[ :, :2])
    ax1.plot(nx.degree_histogram(graph))
    ax1.set_title("Degree histogram")
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("# of Nodes")

    ax2 = fig.add_subplot(axgrid[ :, 2:])
    ax2.loglog(nx.degree_histogram(graph))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes (log log scale)")

    fig.tight_layout()
    if save_path == None:
        plt.show()
    else:
        plt.savefig(f'{save_path}.png')
        plt.close()



    
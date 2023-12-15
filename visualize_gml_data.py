import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class gml_dataset():
    def __init__(self, datapath) -> None:
        self.datapath=datapath
        self.data=nx.read_gml(datapath)
        self.names_index=[(index,name) for index, name in enumerate(self.data.nodes())]

    def print_general(self):
        n_nodes = self.data.number_of_nodes()
        n_edges = self.data.number_of_edges()
        print("The graph has ", n_nodes," nodes and ", n_edges, " edges")
        print("The graph has ", nx.number_connected_components(self.data)," connected components.")
        largest_cc = max(nx.connected_components(self.data), key=len)

        print("The largest connected component in the graph has ", len(largest_cc), 'nodes')

        self.subG = self.data.subgraph(largest_cc)
        print("The largest connected component in the graph has ", self.subG.number_of_edges(), ' edges')
        
        # Degree
        degree_sequence = [self.data.degree(node) for node in self.data.nodes()]

        min_deg = np.min(degree_sequence)
        max_deg = np.max(degree_sequence)
        mean_deg = np.mean(degree_sequence)
        med_deg = np.median(degree_sequence)

        print("The minimum degree of a node is : ", min_deg )
        print("The maximum degree of a node is : ", max_deg )
        print("The mean degree of a node is : ", mean_deg)
        print("The median degree of a node is : ", med_deg )
        
        clustering_coef = nx.transitivity(self.data)
        print("The clustering coefficient of the graph G is : ", clustering_coef)

    def show_graph(self, with_labels=True, node_size = 300, font_size=10):
        # Utilisez l'algorithme de disposition du ressort pour positionner les nœuds
        pos = nx.spring_layout(self.data)

        # Dessinez le graphe en utilisant les positions calculées
        nx.draw(self.data, pos, with_labels=with_labels, node_size=node_size, node_color='skyblue', font_size=font_size, width = 0.1)

        # Affichez le graphe
        plt.show()
    
    def plot_figure_graph(self):
        fig = plt.figure("Degree of the graph", figsize=(8, 4))
        # Create a gridspec for adding subplots of different sizes
        axgrid = fig.add_gridspec(5, 4)

        ax1 = fig.add_subplot(axgrid[ :, :2])
        ax1.plot(nx.degree_histogram(self.data))
        ax1.set_title("Degree histogram")
        ax1.set_xlabel("Degree")
        ax1.set_ylabel("# of Nodes")

        ax2 = fig.add_subplot(axgrid[ :, 2:])
        ax2.loglog(nx.degree_histogram(self.data))
        ax2.set_title("Degree histogram")
        ax2.set_xlabel("Degree")
        ax2.set_ylabel("# of Nodes (log log scale)")

        fig.tight_layout()
        plt.show()

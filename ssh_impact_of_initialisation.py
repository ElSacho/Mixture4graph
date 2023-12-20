import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import vizualize_data as vd

from EM_torch import mixtureModel

import os

from EM_torch import mixtureModel

print("Name of the experiment results folder : " , end="")
save_name = input()
print("Working on it")
# Create the directory
os.makedirs(os.path.join("results",save_name), exist_ok=True)

print("What data do you want to use ? [metro, political, twitter, miserables] : " , end="")
data_name = input()
if data_name == 'metro':
    graph = nx.read_gml("data/metro/final_metro_network.gml")
elif data_name == 'political':
    graph = nx.read_gml("data/political_portrait/political_extraction.gml")
elif data_name == 'twitter':
    graph = nx.read_edgelist("data/congress_network/congress_new_edgelist.txt")
elif data_name == 'miserables':
    graph = nx.read_gml("data/lesmis/lesmis.gml")
else:
    graph = nx.read_gml("data/lesmis/lesmis.gml")
print("Graph uploaded")

vd.print_generalitize(graph)
vd.plot_figure_graph(graph, save_path=os.path.join("results",save_name,'graph_information'))

# We remove the isolated nodes
isolated_nodes = list(nx.isolates(graph))
graph.remove_nodes_from(isolated_nodes)

model = mixtureModel(graph)
tab_clusters = range(2, 11)

model.precise_fit(tab_clusters, nbr_iter_per_cluster=20 , max_iter_em=20, list_initialisation_methods = [ 'modularity', 'spectral','random'])
model.plot_repeated_ICL(save_path=f'all_ICL_{data_name}')

model.plot_jrx(save_path=os.path.join("results",save_name,'jrx_generate'))
model.plot_icl(save_path=os.path.join("results",save_name,'icl_generate'))
model.plot_all_adjency_matrices(save_path=os.path.join("results",save_name,'adjency_matrix'))

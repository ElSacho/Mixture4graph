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

model = mixtureModel(graph, initilisation_method="random")
tab_clusters = range(2, 11)



model.fit(tab_clusters, save_path=os.path.join("results",save_name,'results'))

model.plot_jrx(save_path=os.path.join("results",save_name,'jrx_generate'))
model.plot_icl(save_path=os.path.join("results",save_name,'icl_generate'))
model.plot_all_adjency_matrices(save_path=os.path.join("results",save_name,'adjency_matrix'))

# import numpy as np
# import matplotlib.pyplot as plt

# from utils import show_graph, show_multiple_graph, show_graph_cluster_color
# from generator import generate, generate_and_give_tau

# import os

# from EM_torch import mixtureModel

# print("Name of the experiment results files : " , end="")
# save_name = input()
# print("Working on it")

# np.random.seed(2)
# n_vertices = 500
# pi = np.array([[1,0.1],[0.1,1]])
# priors = np.array([0.2,0.8])
# n_clusters = len(priors)
# max_iter = 50

# graph, tau = generate_and_give_tau(n_vertices, pi , priors)

# model = mixtureModel(graph, initilisation_method='random')
# tab_clusters = range(2, 17)
# model.fit(tab_clusters, save_path=os.path.join("results",'results_'+save_name))

# model.plot_jrx(save_path=os.path.join("results",'jrx_generate'+save_name))
# model.plot_icl(save_path=os.path.join("results",'icl_generate'+save_name))

# # model.plot_adjency_matrix(2, save_path=os.path.join("results",'adjency_matrix'+save_name))
# model.plot_all_adjency_matrices(save_path=os.path.join("results",'adjency_matrix'+save_name))

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from vizualize_data import data_visualisation

from EM_torch import mixtureModel

import os

from EM_torch import mixtureModel

print("Name of the experiment results files : " , end="")
save_name = input()
print("Working on it")

graph = nx.read_gml("data/lesmis/lesmis.gml")
data_visualisation = data_visualisation(graph)

data_visualisation.print_generalitize()
data_visualisation.plot_figure_graph()

model = mixtureModel(graph, initilisation_method="random")
tab_clusters = range(2, 26)


model.fit(tab_clusters, save_path=os.path.join("results",'results_'+save_name))

model.plot_jrx(save_path=os.path.join("results",'jrx_generate'+save_name))
model.plot_icl(save_path=os.path.join("results",'icl_generate'+save_name))
model.plot_all_adjency_matrices(save_path=os.path.join("results",'adjency_matrix'+save_name))


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

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from vizualize_data import dataset

from EM_torch import mixtureModel

data = dataset("data/CA-HepTh.txt")

data.print_generalitize()
data.plot_figure_graph()

model = mixtureModel(data.get_connected_component(), initilisation_method="random")
tab_clusters = [2,3,4,5]
model.fit(tab_clusters)

model.plot_jrx(save_path='jrx_CA')
model.plot_icl(save_path='icl_CA')
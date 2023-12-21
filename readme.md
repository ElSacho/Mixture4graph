# A Mixture Model for Random Graphs

## Abstract
Traditional graph analysis methods predominantly focus on node degree to infer graph structure, typically grouping nodes into classes based on high interaction rates within each class. However, this approach often fail to capture the underlying structure of a graph. In this report, we delve into a mixture model that shifts the focus from nodes to edges, thereby offering a more nuanced depiction of graph structure. Our study emphasizes the importance of parameter initialization in the algorithm and demonstrates its application through the analysis of two real-world datasets. We explore how this edge-centric approach provides a detailed understanding of graph topology, capturing complex structures that node-based methods may miss. However, the effectiveness of this method is somewhat constrained by the graph's size. This limitation highlights the need for further refinement to extend its applicability to larger networks.


## Installation

```bash
git clone https://github.com/ElSacho/Mixture4graph
cd Mixture4graph
pip install -r requirements.txt


## How it Works - The notebook main_notebook.ipynb

You can find most of the following explanations in the notebook file.

- **Generate Graphs** \\
If you want to generate a graph, you need to provide the model with the prior \(\pi\), \(\alpha\), and the number of vertices you want.
You can call the generator with the function `generate that will return the generated graph, or `generate_and_give_tau that will return a tuple with the generated graph, and a vector to explain to which clusters all nodes belong from the generator file.

- **Estimate Parameters**  \\
    - To estimate the parameters of a graph, you first need to initialize a model. To do this, you can call the `mixtureModel` class. \\
    - Then, you can fit the data with the model by calling the `fit` function. You need to specify the number of clusters you want for your fit, and you can also provide the model with a list of number of clusters. You may want to add an initialization method for the first values of \(\tau\), to test the model with several initialization methods.
    - If you want to iterate the algorithme to stabilize the results, you can use the other function `precise_fit that will iterate the fit function for nbr_iter_per_cluster time per cluster, in order to reduce and show the variance of the different methods. As a parameter, you can give a list of method that you want to use, the string argument must be 'modularity', 'spectral','random', ou 'hierarchical'. In our experiments, we only studied the first three of them as the results from 'hierarchical' were not really interesting.

- **Plot the results** \\
    All the functions to plot the results are in the mixtureModel class.  
    - `plotJRX` will plot the values of the $\mathcal{J}(R_{\mathcal(X)}) for all the clusters that has been studied
    - `plotICL` will plot the values of the ICL which is the value that we use to find the good number of clusters
    - `plot_adjency_matrix` will plot the adjency matrices of the final model. If you do not precise any value for the number of clusters, it will plot all the adjency matrix, or just one if you precise the one you want. You may also want to see the labels of the nodes in the adjency matrix, that you can do using using show_names = True

- **Generate a new graph ** \\
    - Finaly, you can generate a new graph with the values of the model. To access those values, you can use the dictionary of the mixtureModel, with the number of clusters you need. It will give you another dictionnary with the returned 'tau', 'pi', 'prior' .. values that you can use. 
    - You can finish ploting the orignal graph and the generated one. To do so, use the function show_colors_multiple_graph, were you give the two graphs, their respective tau values (to add the color on the clusters) and the titles that you want for your two graphs.

## How it Works - Running the code on ssh

You can also run one function of the code that will produce all the results presented in the report. To do so, you can use the file `ssh_initialisation.py. When running this file, you will need to write the name of the output folder in which you want your figures to be saved, and the data that you want to use. A simple reading of the code allows to understand how you can use the created code for your own data : just read your data with nx. 

### VisualizeData
We also created a class `VisualizeData that you can use as `VisualizeData.print_generalities(graph)` to print the most interesting information of the graph, or `VisualizeData.plot_figure_graph(graph)` to plot the degree distribution of the graph. 


## Contribuer
Some contributions of the project could include the implementation of new clustering initialization method, in the file `initialisation_methods.py , or other criterion to choose the number of clusters such as BIC, AIC or anything else.
import networkx as nx

# Charger les fichiers GML
graphs = []
avoid = [7,10,13]
for k in range(1, 15):
    if not k in avoid:
        graphs.append(nx.read_gml(f"data/metro/ligne_per_line/metro_network_line_{k}.gml"))

# Fusionner les graphes
combined_graph = nx.Graph()
for graph in graphs:
    for node, data in graph.nodes(data=True):
        # Si le nœud existe déjà, fusionner les données ou choisir un ensemble de données
        if combined_graph.has_node(node):
            # Choisissez ici comment gérer les données en double
            continue
        combined_graph.add_node(node, **data)
    
    # Ajouter les arêtes
    combined_graph.add_edges_from(graph.edges(data=True))

# Exporter le graphique fusionné
nx.write_gml(combined_graph, "combined_metro_network.gml")

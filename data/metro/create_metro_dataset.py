import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
import json

import networkx as nx
from scipy.spatial import distance
import numpy as np

from unidecode import unidecode

metro = pd.read_csv('data/metro/emplacement-des-gares-idf.csv', sep=';')
metro['geo_shape'] = metro['geo_shape'].apply(json.loads)

ligne_to_avoid = [7, 10, 13]

for k in range(1, 15):
    if not k in ligne_to_avoid:     
        df_metro = metro[ metro['res_com'] == f"METRO {k}"]
        # df_metro['geo_shape'].apply(json.loads)
        # Création d'un graph vide
        G = nx.Graph()

        for index, row in df_metro.iterrows():
            # Extraction directe des coordonnées du dictionnaire 'geo_shape'
            line = row['nom_gares']
            line = unidecode(line)
            line = line.lower()
            # Remplacer les tirets par des espaces
            line = line.replace('-', ' ')
            # Enlever les espaces à la fin des lignes
            line = line.strip()
            # Supprimer les doubles espaces
            clean_line = ' '.join(line.split())
            G.add_node(clean_line, pos=tuple(row['geo_shape']['coordinates']))

        # Lecture et traitement du fichier texte
        with open('metro_clean.txt', 'r') as file:
            lines = file.readlines()

        stations = []
        begin = False
        for line in lines:
            line = line.strip()
            if line.startswith('#'):  # Début d'une nouvelle section
                begin = (line == f"# ligne {k}")
            elif begin and line:  # Collecter les stations si nous sommes dans la bonne section
                clean_line = ' '.join(line.lower().replace('-', ' ').split())
                stations.append(clean_line)

        # Ajout des arêtes
        for i in range(len(stations) - 1):
            start_station = stations[i]
            end_station = stations[i + 1]
            # Ici, vous pouvez ajouter une logique pour gérer les noms non identiques
            G.add_edge(start_station, end_station)
            
        # Identifier les nœuds sans l'attribut 'pos'
        nodes_to_remove = [node for node, attrs in G.nodes(data=True) if 'pos' not in attrs]

        # Supprimer les nœuds identifiés et leurs liens (edges)
        G.remove_nodes_from(nodes_to_remove)
        
        # Recherche des nœuds isolés
        for node in G.nodes:
            if G.degree(node) == 0:
                print("Nœud isolé:", node, 'in ligne ', k)

        # Enregistrement du graph dans un fichier GML
        nx.write_gml(G, f"data/metro/ligne_per_line/metro_network_line_{k}.gml")

    

existing_file_path = 'data/congress_network/congress.edgelist'
new_file_path = 'data/congress_network/congress_new_edgelist.txt'

# Lire le fichier existant
with open(existing_file_path, 'r') as file:
    lines = file.readlines()

# Modifier les lignes pour enlever la partie avec 'weight'
modified_lines = [line.split(' {')[0] + '\n' for line in lines]

# Écrire les nouvelles lignes dans le nouveau fichier
with open(new_file_path, 'w') as new_file:
    new_file.writelines(modified_lines)

result = f"Le fichier a été modifié avec succès. Le nouveau fichier est disponible ici: {new_file_path}"
print(result)

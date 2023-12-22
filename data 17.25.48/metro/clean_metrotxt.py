from unidecode import unidecode
# Lecture du fichier original
file_path = 'metro.txt'

# Lecture du contenu du fichier
with open(file_path, 'r') as file:
    lines = file.readlines()

# Nettoyage des lignes
clean_lines = []
for line in lines:
    line = unidecode(line)
    # Convertir en minuscules
    line = line.lower()
    # Remplacer les tirets par des espaces
    line = line.replace('-', ' ')
    # Enlever les espaces à la fin des lignes
    line = line.strip()
    # Supprimer les doubles espaces
    line = ' '.join(line.split())
    clean_lines.append(line)

# Chemin du fichier de sortie
output_path = 'metro_clean.txt'

# Écriture dans le fichier de sortie
with open(output_path, 'w') as file:
    for line in clean_lines:
        file.write(line + '\n')


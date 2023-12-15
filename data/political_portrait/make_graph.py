from collections import defaultdict, Counter
import string
import networkx as nx
import re

import nltk
from nltk.corpus import stopwords



# Function to remove punctuation from a sentence
def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

# Function to read the file and preprocess the text
# Modified function to read the file and preprocess the text, excluding words with accents or special characters
def read_and_preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Split into sentences
    sentences = text.split('.')
    # Remove punctuation, convert to lowercase, and exclude words with non-alphanumeric characters
    sentences = [re.sub(r'\b\w*[^a-zA-Z0-9\s]+\w*\b', '', sentence) for sentence in sentences]
    sentences = [remove_punctuation(sentence).lower() for sentence in sentences]
    return sentences

# Function to build the graph
def build_graph(sentences):
    # List of common words to ignore
    # common_words = set(['le', 'la', 'les', 'un', 'une', 'des', 'et', 'de', 'du', 'en', 'à', 'pour', 'avec', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'va', 'est'])
    # Downloading the stopwords from NLTK
    nltk.download('stopwords')

    # Getting French stopwords
    french_stopwords = set(stopwords.words('french'))

    # Adding more common words to the set
    additional_common_words = {'cela', 'cette', 'ceux', 'aussi', 'comme', 'mais', 'ou', 'où', 'par', 'pas', 'qui', 'quoi', 'sans', 'sont', 'sous', 'avec'}
    common_words = french_stopwords.union(additional_common_words)
    
    # Counter for word frequencies
    word_freq = Counter()

    # Default dictionary to store connections between words
    connections = defaultdict(set)

    # Processing each sentence
    for sentence in sentences:
        words = sentence.split()
        # Update word frequencies
        word_freq.update(words)
        # Build connections
        for word in words:
            if word not in common_words and len(word) >= 3:
                connections[word].update([w for w in words if w != word and w not in common_words])

    # Removing words that appear only once
    connections = {word: neighbors for word, neighbors in connections.items() if word_freq[word] > 1}

    # Creating a graph
    G = nx.Graph()
    for word, neighbors in connections.items():
        for neighbor in neighbors:
            G.add_edge(word, neighbor)

    return G

# File path
file_path = 'data/political_portrait/political.txt'

# Read and preprocess the file
sentences = read_and_preprocess(file_path)

# Build the graph
graph = build_graph(sentences)

# Save the graph in GML format
output_path = 'data/political_portrait/political_extraction.gml'
nx.write_gml(graph, output_path)

print("finished")

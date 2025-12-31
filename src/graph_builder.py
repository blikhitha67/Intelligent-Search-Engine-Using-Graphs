import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def build_document_graph(tfidf_matrix, threshold=0.2):
    """
    Build a document similarity graph using adjacency list.
    Edge added if similarity > threshold.
    """
    sim_matrix = cosine_similarity(tfidf_matrix)
    G = nx.Graph()
    n = tfidf_matrix.shape[0]
    for i in range(n):
        G.add_node(i)
        for j in range(i+1, n):
            if sim_matrix[i,j] > threshold:
                G.add_edge(i, j, weight=sim_matrix[i,j])
    return G
  
#How It Works
  
# Builds a graph of documents:
# - Each node = document
# - Edges = similarity between documents above a threshold
# - Stored as adjacency list

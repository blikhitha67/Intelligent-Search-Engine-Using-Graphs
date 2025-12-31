import networkx as nx

def compute_pagerank(graph, alpha=0.85):
    """
    Compute PageRank scores for documents
    """
    return nx.pagerank(graph, alpha=alpha)

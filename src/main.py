from preprocessing import clean_text
from vectorization import vectorize_documents
from graph_builder import build_document_graph
from pagerank import compute_pagerank
from search import search_query

# Example documents
docs = [
    "Python and AI for cybersecurity.",
    "Network security and firewall configuration.",
    "Machine learning for threat detection.",
    "Frontend development using React and JavaScript."
]

# Preprocess
docs_clean = [clean_text(d) for d in docs]

# Vectorize
vectorizer, tfidf_matrix = vectorize_documents(docs_clean)

# Build graph
graph = build_document_graph(tfidf_matrix)

# Compute PageRank
pagerank_scores = compute_pagerank(graph)

# Run search
query = "AI cybersecurity"
results = search_query(query, docs, tfidf_matrix, vectorizer, pagerank_scores)

print("Top results for query:", query)
for doc_idx, score in results:
    print(f"Doc {doc_idx}: {docs[doc_idx]} (Score: {score:.2f})")

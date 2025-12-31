from preprocessing import clean_text
from sklearn.metrics.pairwise import cosine_similarity

def search_query(query, docs, tfidf_matrix, vectorizer, pagerank_scores, top_n=5):
    """
    Search documents using query + PageRank
    """
    query_clean = clean_text(query)
    query_vec = vectorizer.transform([query_clean])
    sim_scores = cosine_similarity(tfidf_matrix, query_vec).flatten()

    # Combine similarity + PageRank
    combined_scores = {i: sim_scores[i] + pagerank_scores[i] for i in range(len(docs))}
    # Sort top_n
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return ranked

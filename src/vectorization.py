from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_documents(docs):
    """
    Convert documents to TF-IDF vectors.
    Returns vectorizer and tfidf_matrix
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer, tfidf_matrix

#How It Works

# Converts text into numerical representations:
# - TF-IDF or embeddings
# - Used for similarity scoring between query and documents

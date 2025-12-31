import re
import spacy

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Cleans and normalizes text.
    - Lowercase
    - Remove special characters
    - Tokenize & lemmatize
    - Remove stopwords
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

#How It Works

# Cleans and normalizes document text:
# - Lowercase conversion
# - Remove special characters
# - Tokenization and lemmatization
# - Stopword removal

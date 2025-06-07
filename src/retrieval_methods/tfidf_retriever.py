from src.indexing.inverted_index_builder import InvertedIndex
from src.indexing.text_processor import TextProcessor
import math
import numpy as np
from collections import Counter


class TFIDFRetriever:
    """TF-IDF based retrieval system"""

    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index
        self.tf_idf_cache = {}

    def calculate_tf(self, term, doc_id):
        """Calculate term frequency"""
        doc = self.index.documents[doc_id]
        text_content = f"{doc.get('Title', '')} {doc.get('Script', '')}"
        tokens = TextProcessor.preprocess(text_content)
        term_count = tokens.count(term)
        return term_count / len(tokens) if tokens else 0

    def calculate_idf(self, term):
        """Calculate inverse document frequency"""
        if term not in self.index.index:
            return 0

        df = len(self.index.index[term])  # Document frequency
        return math.log(self.index.total_docs / df) if df > 0 else 0

    def calculate_tf_idf(self, term, doc_id):
        """Calculate TF-IDF score"""
        tf = self.calculate_tf(term, doc_id)
        idf = self.calculate_idf(term)
        return tf * idf

    def get_document_vector(self, doc_id, query_terms):
        """Get TF-IDF vector for a document"""
        vector = []
        for term in query_terms:
            vector.append(self.calculate_tf_idf(term, doc_id))
        return np.array(vector)

    def get_query_vector(self, query_terms):
        """Get TF-IDF vector for query"""
        term_counts = Counter(query_terms)
        vector = []
        for term in query_terms:
            tf = term_counts[term] / len(query_terms)
            idf = self.calculate_idf(term)
            vector.append(tf * idf)
        return np.array(vector)

    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query, top_k=10):
        """Perform TF-IDF based search"""
        query_terms = TextProcessor.preprocess(query)  # Remove duplicates
        if not query_terms:
            return []

        # Get all candidate documents
        candidate_docs = set()
        for term in query_terms:
            if term in self.index.index:
                candidate_docs.update(self.index.index[term])

        if not candidate_docs:
            print("No documents matched the query terms.")
            return []

        # Calculate similarities
        query_vector = self.get_query_vector(query_terms)
        similarities = []

        for doc_id in candidate_docs:
            doc_vector = self.get_document_vector(doc_id, query_terms)
            similarity = self.cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                similarities.append((doc_id, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
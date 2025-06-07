import math
from src.indexing.text_processor import TextProcessor
from src.indexing.inverted_index_builder import InvertedIndex


class BM25Retriever:
    """BM25 retrieval system"""

    def __init__(self, inverted_index: InvertedIndex, k1=1.5, b=0.75):
        self.index = inverted_index
        self.k1 = k1
        self.b = b

    def calculate_bm25_score(self, query_terms, doc_id):
        """Calculate BM25 score for a document given query terms"""
        doc = self.index.documents[doc_id]
        text_content = f"{doc.get('Title', '')} {doc.get('Script', '')}"
        doc_tokens = TextProcessor.preprocess(text_content)
        doc_length = len(doc_tokens)

        score = 0
        for term in query_terms:
            if term in self.index.index and doc_id in self.index.index[term]:
                tf = doc_tokens.count(term)
                df = len(self.index.index[term])
                idf = math.log((self.index.total_docs - df + 0.5) / (df + 0.5))
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.index.avg_doc_length))
                score += idf * (numerator / denominator)

        return score

    def search(self, query, top_k=10):
        """Perform BM25 based search"""
        query_terms = TextProcessor.preprocess(query)
        if not query_terms:
            return []

        candidate_docs = set()
        for term in query_terms:
            if term in self.index.index:
                candidate_docs.update(self.index.index[term])

        if not candidate_docs:
            return []

        scores = []
        for doc_id in candidate_docs:
            score = self.calculate_bm25_score(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
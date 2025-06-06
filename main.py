# src/indexing/inverted_index_builder.py

import pickle
from collections import defaultdict
from src.indexing.text_processor import TextProcessor

class InvertedIndex:
    """Handles inverted index creation and basic search"""

    def __init__(self):
        self.index = defaultdict(set)
        self.documents = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0

    def build_index(self, dataset, use_bigrams=False):
        """Build inverted index from dataset"""
        self.documents = {doc['id']: doc for doc in dataset}
        self.total_docs = len(dataset)
        total_length = 0

        for episode in dataset:
            doc_id = episode['id']
            text_content = f"{episode.get('Title', '')} {episode.get('Script', '')}"

            if use_bigrams:
                tokens = TextProcessor.preprocess_with_bigrams(text_content)
            else:
                tokens = TextProcessor.preprocess(text_content)

            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            for token in tokens:
                self.index[token].add(doc_id)

        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        print(f"✅ Built inverted index with {len(self.index)} unique terms")

    def boolean_search(self, query):
        """Basic boolean search (AND operation)"""
        query_words = TextProcessor.preprocess(query)
        if not query_words:
            return []

        result_sets = [self.index.get(word, set()) for word in query_words]
        result_ids = set.intersection(*result_sets) if result_sets else set()
        return sorted(result_ids)

    def save_index(self, filename="inverted_index.pkl"):
        """Save inverted index to file"""
        index_data = {
            'index': dict(self.index),
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'total_docs': self.total_docs
        }

        with open(filename, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"✅ Saved inverted index to {filename}")

    def load_index(self, filename="inverted_index.pkl"):
        """Load inverted index from file"""
        with open(filename, 'rb') as f:
            index_data = pickle.load(f)

        self.index = defaultdict(set, index_data['index'])
        self.documents = index_data['documents']
        self.doc_lengths = index_data['doc_lengths']
        self.avg_doc_length = index_data['avg_doc_length']
        self.total_docs = index_data['total_docs']
        print(f"✅ Loaded inverted index from {filename}")

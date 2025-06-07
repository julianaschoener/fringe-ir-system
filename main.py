# src/indexing/inverted_index_builder.py
# Fringe Information Retrieval System
# Complete implementation with all retrieval methods and evaluation

import os
import re
import time
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import requests
from bs4 import BeautifulSoup

import json
import pickle
from collections import defaultdict
from src.indexing.text_processor import TextProcessor
from src.indexing.inverted_index_builder import InvertedIndex
from src.retrieval_methods.tfidf_retriever import TFIDFRetriever
from src.retrieval_methods.semantic_retriever import SemanticRetriever
from src.retrieval_methods.bm25_retriever import BM25Retriever
import pandas as pd
from src.evaluation.ndcg_evaluator import NDCGEvaluator
from src.evaluation.utils import load_golden_dataset_from_csv


def main():
# 1. Load dataset
    with open("data/fringe_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        doc_map = {doc['id']: doc for doc in dataset}

    # 2. Build the inverted index
    index = InvertedIndex()
    index.build_index(dataset)
    index.save_index("inverted_index.pkl")
    index.load_index("inverted_index.pkl")


    # 2. Optionally initialize the semantic retriever
    semantic_retriever = SemanticRetriever(index)
    # 3. Load semantic embeddings if available
    print("Building embeddings...")
    semantic_retriever.build_embeddings()
    semantic_retriever.save_embeddings()
    semantic_retriever.load_embeddings("semantic_embeddings.pkl")


    query = input("🔍 Enter your query: ")

    bm25 = BM25Retriever(index)
    bm25_results = bm25.search(query)
    print("📘 BM25 Top Results:")
    for doc_id, score in bm25_results[:5]:
        doc = doc_map.get(doc_id)
        print(f"Doc ID: {doc_id} | Title: {doc['Title']} | Score: {score:.4f}")

    bool_results = index.boolean_search(query)
    print("📘 Boolean Search:")
    for doc_id in bool_results:
        doc = doc_map.get(doc_id)
        print(f"Doc ID: {doc_id} | Title: {doc['Title']}")

    retriever = TFIDFRetriever(index)
    tfidf_results = retriever.search(query)
    print("📘 TF-IDF Top Results:")
    for doc_id, score in tfidf_results[:5]:
        doc = doc_map.get(doc_id)
        snippet = doc['Script'][:200].replace("\n", " ")  # nur ein Auszug
        print(f"Doc ID: {doc_id} | Title: {doc['Title']} | Score: {score:.4f}")
        print(f"Snippet: {snippet}...\n")

    semantic_results = semantic_retriever.search(query, top_k=5)
    print("📘 Semantic Search Top:")
    for doc_id, score in semantic_results:
        doc = doc_map.get(doc_id)
        snippet = doc['Script'][:200].replace("\n", " ")
        print(f"Doc ID: {doc_id} | Title: {doc['Title']} | Score: {score:.4f}")
        print(f"Snippet: {snippet}...\n")






    golden_dataset = load_golden_dataset_from_csv()

    retrieval_results = {}
    for item in golden_dataset:
        test_query = item['query']
        test_results = bm25.search(test_query)
        retrieval_results[test_query] = test_results

    score = NDCGEvaluator.evaluate_retrieval(golden_dataset, retrieval_results, k=10)
    print(f"nDCG@10: {score:.4f}")

    print(f"🔍 Total queries processed: {len(retrieval_results)}")
    for query, results in list(retrieval_results.items())[:2]:  # Erste 2 Queries
        print(f"Query: '{query}' -> Found {len(results)} results")
        if results:
            print(f"  Top result: Doc {results[0][0]} (Score: {results[0][1]:.4f})")

class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(set)
        self.documents = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0

    def build_index(self, dataset, use_bigrams=False):
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
        query_words = TextProcessor.preprocess(query)
        if not query_words:
            return []

        result_sets = [self.index.get(word, set()) for word in query_words]
        result_ids = set.intersection(*result_sets) if result_sets else set()
        return sorted(result_ids)

    def save_index(self, filename="inverted_index.pkl"):
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
        with open(filename, 'rb') as f:
            index_data = pickle.load(f)

        self.index = defaultdict(set, index_data['index'])
        self.documents = index_data['documents']
        self.doc_lengths = index_data['doc_lengths']
        self.avg_doc_length = index_data['avg_doc_length']
        self.total_docs = index_data['total_docs']
        print(f"✅ Loaded inverted index from {filename}")




if __name__ == "__main__":
    main()

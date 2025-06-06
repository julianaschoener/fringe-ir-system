# Fringe Information Retrieval System
# Complete implementation with all retrieval methods and evaluation

import os
import re
import json
import pickle
import time
import math
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import requests
from bs4 import BeautifulSoup

# For semantic similarity (you'll need to install these)
try:
    from sentence_transformers import SentenceTransformer
    import faiss

    SEMANTIC_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers and/or faiss not available. Semantic search will be disabled.")
    SEMANTIC_AVAILABLE = False



def main():
    # Step 1: Scrape or load the dataset
    # Step 2: Build the inverted index
    # Step 3: Run an example search
    pass  # fill in

if __name__ == "__main__":
    main()
from src.indexing.inverted_index_builder import InvertedIndex
from src.retrieval_methods.tfidf_retriever import TFIDFRetriever

# build/load index
index = InvertedIndex()
index.load_index("inverted_index.pkl")

# use TF-IDF retrieval
retriever = TFIDFRetriever(index)
results = retriever.search("Walter Bishop experiments", top_k=5)


from src.indexing.inverted_index_builder import InvertedIndex
from src.retrieval_methods.bm25_retriever import BM25Retriever

index = InvertedIndex()
index.load_index("inverted_index.pkl")

retriever = BM25Retriever(index)
results = retriever.search("Peter Bishop cortexiphan", top_k=5)

from src.indexing.text_preprocessor import TextPreprocessor
from src.indexing.inverted_index_builder import InvertedIndex

from src.indexing.inverted_index_builder import InvertedIndex
from src.retrieval_methods.semantic_retriever import SemanticRetriever

index = InvertedIndex()
index.load_index("inverted_index.pkl")

retriever = SemanticRetriever(index)
retriever.build_embeddings()  # oder retriever.load_embeddings()
results = retriever.search("parallel universes and Olivia Dunham", top_k=5)


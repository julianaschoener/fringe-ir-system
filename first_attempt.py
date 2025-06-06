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

class FringeDataProcessor:
    """Handles data scraping and preprocessing"""

    def __init__(self):
        self.dataset = []
        self.episode_id = 1

    def clean_text(self, text):
        """Clean and normalize text"""
        return ' '.join(text.strip().split())

    def extract_season_episode(self, page_text):
        """Extract season and episode numbers"""
        season = "Unknown"
        episode = "Unknown"

        match = re.search(r"Season\s+(\d+)[,\s]+Episode\s+(\d+)", page_text, re.IGNORECASE)
        if match:
            season = match.group(1)
            episode = match.group(2)
        else:
            match = re.search(r"S(\d+)E(\d+)", page_text, re.IGNORECASE)
            if match:
                season = match.group(1)
                episode = match.group(2)

        return season, episode

    def extract_writers_directors(self, page_text):
        """Extract writers and directors information"""
        writers = "Unknown"
        directors = "Unknown"

        writer_match = re.search(r'Written by[:\s]+([^\n]+)', page_text, re.IGNORECASE)
        director_match = re.search(r'Directed by[:\s]+([^\n]+)', page_text, re.IGNORECASE)

        if writer_match:
            writers = self.clean_text(writer_match.group(1))
        if director_match:
            directors = self.clean_text(director_match.group(1))

        return writers, directors

    def extract_cast(self, page_text):
        """Extract cast information"""
        cast_list = []
        matches = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+as\s+([A-Z][a-zA-Z\s]+)', page_text)
        for actor, character in matches:
            cast_list.append(f"{actor} as {character}")
        return list(set(cast_list))

    def extract_script(self, transcript_soup):
        """Extract script text from transcript page"""
        script_text = ""
        content_div = transcript_soup.find("div", {"class": "mw-parser-output"})
        if content_div:
            paragraphs = content_div.find_all(["p", "ul", "ol"])
            for tag in paragraphs:
                script_text += self.clean_text(tag.text) + "\n"
        return script_text.strip()

    def scrape_episodes(self, urls):
        """Scrape all episodes from provided URLs"""
        self.dataset = []
        self.episode_id = 1

        for episode_url, transcript_url, cast_url in urls:
            print(f"Scraping episode {self.episode_id}: {episode_url.split('/')[-1]}...")
            try:
                ep_page = requests.get(episode_url)
                tr_page = requests.get(transcript_url)
                cast_page = requests.get(cast_url)

                episode_soup = BeautifulSoup(ep_page.content, "html.parser")
                transcript_soup = BeautifulSoup(tr_page.content, "html.parser")
                cast_soup = BeautifulSoup(cast_page.content, "html.parser")

                ep_text = episode_soup.get_text(separator='\n')
                cast_text = cast_soup.get_text(separator='\n')

                writers, directors = self.extract_writers_directors(ep_text)
                season, episode = self.extract_season_episode(ep_text)
                cast = self.extract_cast(cast_text)

                title_tag = episode_soup.find("h1", id="firstHeading")
                title = self.clean_text(title_tag.text) if title_tag else "Unknown Title"

                script = self.extract_script(transcript_soup)

                self.dataset.append({
                    "id": self.episode_id,
                    "Title": title,
                    "Season": season,
                    "Episode": episode,
                    "Writers": writers,
                    "Directors": directors,
                    "Cast": cast,
                    "Script": script
                })
                self.episode_id += 1
                time.sleep(1)  # Be respectful to the server

            except Exception as e:
                print(f"Failed scraping {episode_url}: {e}")

        return self.dataset

    def save_dataset(self, filename_base="fringe_dataset"):
        """Save dataset in multiple formats"""
        df = pd.DataFrame(self.dataset)

        # Save as CSV
        df.to_csv(f"{filename_base}.csv", index=False)
        print(f"✅ Saved dataset to {filename_base}.csv")

        # Save as JSON
        df.to_json(f"{filename_base}.json", orient="records", indent=2)
        print(f"✅ Saved dataset to {filename_base}.json")

        # Save as pickle
        with open(f'{filename_base}.pkl', 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f"✅ Saved dataset to {filename_base}.pkl")

        return df

class TextPreprocessor:
    """Handles text preprocessing and tokenization"""

    @staticmethod
    def preprocess(text):
        """Basic text preprocessing"""
        if not text:
            return []
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        return [word for word in words if len(word) > 1]  # Remove single characters

    @staticmethod
    def preprocess_with_bigrams(text):
        """Preprocessing with bigram support"""
        words = TextPreprocessor.preprocess(text)
        bigrams = []
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]}_{words[i+1]}")
        return words + bigrams

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

            # Combine relevant text fields
            text_content = f"{episode.get('Title', '')} {episode.get('Script', '')}"

            if use_bigrams:
                tokens = TextPreprocessor.preprocess_with_bigrams(text_content)
            else:
                tokens = TextPreprocessor.preprocess(text_content)

            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            # Build inverted index
            for token in tokens:
                self.index[token].add(doc_id)

        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        print(f"✅ Built inverted index with {len(self.index)} unique terms")

    def boolean_search(self, query):
        """Basic boolean search (AND operation)"""
        query_words = TextPreprocessor.preprocess(query)
        if not query_words:
            return []

        result_sets = []
        for word in query_words:
            if word in self.index:
                result_sets.append(set(self.index[word]))
            else:
                result_sets.append(set())

        if result_sets:
            result_ids = set.intersection(*result_sets)
        else:
            result_ids = set()

        return sorted(result_ids)

    def save_index(self, filename="inverted_index.pkl"):
        """Save inverted index to file"""
        index_data = {
            'index': dict(self.index),  # Convert defaultdict to dict for pickling
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

class TFIDFRetriever:
    """TF-IDF based retrieval system"""

    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index
        self.tf_idf_cache = {}

    def calculate_tf(self, term, doc_id):
        """Calculate term frequency"""
        doc = self.index.documents[doc_id]
        text_content = f"{doc.get('Title', '')} {doc.get('Script', '')}"
        tokens = TextPreprocessor.preprocess(text_content)
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
        query_terms = list(set(TextPreprocessor.preprocess(query)))  # Remove duplicates
        if not query_terms:
            return []

        # Get all candidate documents
        candidate_docs = set()
        for term in query_terms:
            if term in self.index.index:
                candidate_docs.update(self.index.index[term])

        if not candidate_docs:
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
        doc_tokens = TextPreprocessor.preprocess(text_content)
        doc_length = len(doc_tokens)

        score = 0
        for term in query_terms:
            if term in self.index.index and doc_id in self.index.index[term]:
                # Term frequency in document
                tf = doc_tokens.count(term)

                # Document frequency
                df = len(self.index.index[term])

                # IDF component
                idf = math.log((self.index.total_docs - df + 0.5) / (df + 0.5))

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.index.avg_doc_length))

                score += idf * (numerator / denominator)

        return score

    def search(self, query, top_k=10):
        """Perform BM25 based search"""
        query_terms = TextPreprocessor.preprocess(query)
        if not query_terms:
            return []

        # Get all candidate documents
        candidate_docs = set()
        for term in query_terms:
            if term in self.index.index:
                candidate_docs.update(self.index.index[term])

        if not candidate_docs:
            return []

        # Calculate BM25 scores
        scores = []
        for doc_id in candidate_docs:
            score = self.calculate_bm25_score(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class SemanticRetriever:
    """Semantic similarity based retrieval using sentence transformers"""

    def __init__(self, inverted_index: InvertedIndex, model_name='paraphrase-MiniLM-L6-v2'):
        self.index = inverted_index
        self.model_name = model_name
        self.model = None
        self.doc_embeddings = None
        self.faiss_index = None

        if SEMANTIC_AVAILABLE:
            self.model = SentenceTransformer(model_name)
        else:
            print("Warning: Semantic retrieval not available. Install sentence-transformers and faiss.")

    def build_embeddings(self):
        """Build embeddings for all documents"""
        if not SEMANTIC_AVAILABLE:
            print("Error: sentence-transformers not available")
            return

        print("Building document embeddings...")
        texts = []
        doc_ids = []

        for doc_id, doc in self.index.documents.items():
            text_content = f"{doc.get('Title', '')} {doc.get('Script', '')}"
            texts.append(text_content)
            doc_ids.append(doc_id)

        # Generate embeddings
        self.doc_embeddings = self.model.encode(texts)
        self.doc_ids = doc_ids

        # Build FAISS index for fast similarity search
        dimension = self.doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.doc_embeddings)
        self.faiss_index.add(self.doc_embeddings)

        print(f"✅ Built semantic embeddings for {len(texts)} documents")

    def search(self, query, top_k=10):
        """Perform semantic similarity search"""
        if not SEMANTIC_AVAILABLE or self.doc_embeddings is None:
            print("Error: Semantic search not available or embeddings not built")
            return []

        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search using FAISS
        similarities, indices = self.faiss_index.search(query_embedding, top_k)

        # Convert results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.doc_ids):  # Valid index
                doc_id = self.doc_ids[idx]
                results.append((doc_id, float(similarity)))

        return results

    def save_embeddings(self, filename="semantic_embeddings.pkl"):
        """Save embeddings to file"""
        if self.doc_embeddings is not None:
            embedding_data = {
                'embeddings': self.doc_embeddings,
                'doc_ids': self.doc_ids,
                'model_name': self.model_name
            }
            with open(filename, 'wb') as f:
                pickle.dump(embedding_data, f)
            print(f"✅ Saved semantic embeddings to {filename}")

    def load_embeddings(self, filename="semantic_embeddings.pkl"):
        """Load embeddings from file"""
        try:
            with open(filename, 'rb') as f:
                embedding_data = pickle.load(f)

            self.doc_embeddings = embedding_data['embeddings']
            self.doc_ids = embedding_data['doc_ids']

            # Rebuild FAISS index
            if SEMANTIC_AVAILABLE:
                dimension = self.doc_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(self.doc_embeddings)
                print(f"✅ Loaded semantic embeddings from {filename}")

        except FileNotFoundError:
            print(f"Embeddings file {filename} not found. Please build embeddings first.")

class NDCGEvaluator:
    """nDCG evaluation for retrieval systems"""

    @staticmethod
    def dcg_at_k(relevance_scores, k):
        """Calculate DCG@k"""
        relevance_scores = relevance_scores[:k]
        dcg = relevance_scores[0] if len(relevance_scores) > 0 else 0
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / math.log2(i + 1)
        return dcg

    @staticmethod
    def ndcg_at_k(predicted_relevance, ideal_relevance, k):
        """Calculate nDCG@k"""
        dcg = NDCGEvaluator.dcg_at_k(predicted_relevance, k)
        idcg = NDCGEvaluator.dcg_at_k(sorted(ideal_relevance, reverse=True), k)
        return dcg / idcg if idcg > 0 else 0

    @staticmethod
    def evaluate_retrieval(golden_dataset, retrieval_results, k=10):
        """Evaluate retrieval system using nDCG"""
        ndcg_scores = []

        for query_data in golden_dataset:
            query = query_data['query']
            expected_results = {item['episode_id']: item['relevance']
                              for item in query_data['expected_results']}

            if query in retrieval_results:
                predicted_ranking = retrieval_results[query]

                # Extract relevance scores for predicted ranking
                predicted_relevance = []
                for doc_id, _ in predicted_ranking[:k]:
                    relevance = expected_results.get(doc_id, 0)
                    predicted_relevance.append(relevance)

                # Get ideal relevance scores
                ideal_relevance = list(expected_results.values())

                # Calculate nDCG
                ndcg = NDCGEvaluator.ndcg_at_k(predicted_relevance, ideal_relevance, k)
                ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0

class FringeIRSystem:
    """Main Information Retrieval System"""

    def __init__(self):
        self.data_processor = FringeDataProcessor()
        self.inverted_index = InvertedIndex()
        self.tfidf_retriever = None
        self.bm25_retriever = None
        self.semantic_retriever = None
        self.evaluator = NDCGEvaluator()

    def build_system(self, dataset=None, urls=None):
        """Build the complete IR system"""
        # Step 1: Load or scrape data
        if dataset is None and urls is not None:
            print("🔄 Scraping episode data...")
            dataset = self.data_processor.scrape_episodes(urls)
            self.data_processor.save_dataset()
        elif dataset is None:
            print("❌ No dataset or URLs provided")
            return
        else:
            self.data_processor.dataset = dataset

        # Step 2: Build inverted index
        print("🔄 Building inverted index...")
        self.inverted_index.build_index(self.data_processor.dataset)
        self.inverted_index.save_index()

        # Step 3: Initialize retrievers
        print("🔄 Initializing retrieval systems...")
        self.tfidf_retriever = TFIDFRetriever(self.inverted_index)
        self.bm25_retriever = BM25Retriever(self.inverted_index)
        self.semantic_retriever = SemanticRetriever(self.inverted_index)

        # Step 4: Build semantic embeddings if available
        if SEMANTIC_AVAILABLE:
            print("🔄 Building semantic embeddings...")
            self.semantic_retriever.build_embeddings()
            self.semantic_retriever.save_embeddings()

        print("✅ Fringe IR System built successfully!")

    def search_all_methods(self, query, top_k=10):
        """Search using all available methods"""
        results = {}

        # Boolean search
        boolean_results = self.inverted_index.boolean_search(query)
        results['Boolean'] = [(doc_id, 1.0) for doc_id in boolean_results[:top_k]]

        # TF-IDF search
        if self.tfidf_retriever:
            results['TF-IDF'] = self.tfidf_retriever.search(query, top_k)

        # BM25 search
        if self.bm25_retriever:
            results['BM25'] = self.bm25_retriever.search(query, top_k)

        # Semantic search
        if self.semantic_retriever and SEMANTIC_AVAILABLE:
            try:
                results['Semantic'] = self.semantic_retriever.search(query, top_k)
            except:
                print("Warning: Semantic search failed")

        return results

    def evaluate_system(self, golden_dataset_file, k=10):
        """Evaluate all retrieval methods"""
        # Load golden dataset
        try:
            with open(golden_dataset_file, 'r') as f:
                golden_dataset = json.load(f)
        except FileNotFoundError:
            print(f"Golden dataset file {golden_dataset_file} not found")
            return {}

        # Collect results for all queries
        all_results = {
            'Boolean': {},
            'TF-IDF': {},
            'BM25': {},
            'Semantic': {}
        }

        for query_data in golden_dataset:
            query = query_data['query']
            search_results = self.search_all_methods(query, k)

            for method, results in search_results.items():
                all_results[method][query] = results

        # Evaluate each method
        evaluation_results = {}
        for method, method_results in all_results.items():
            if method_results:  # Only evaluate if results exist
                ndcg_score = self.evaluator.evaluate_retrieval(
                    golden_dataset, method_results, k
                )
                evaluation_results[method] = ndcg_score

        return evaluation_results

    def display_results(self, query, results):
        """Display search results in a nice format"""
        print(f"\n🔍 Query: '{query}'")
        print("=" * 60)

        for method, method_results in results.items():
            print(f"\n📊 {method} Results:")
            print("-" * 30)

            if not method_results:
                print("No results found")
                continue

            for rank, (doc_id, score) in enumerate(method_results[:5], 1):
                doc = self.inverted_index.documents.get(doc_id, {})
                title = doc.get('Title', 'Unknown')
                season = doc.get('Season', 'Unknown')
                episode = doc.get('Episode', 'Unknown')

                print(f"{rank}. {title} (S{season}E{episode}) - Score: {score:.4f}")

# Example usage and demo functions
def create_sample_golden_dataset():
    """Create a sample golden dataset for testing"""
    sample_golden = [
        {
            "query": "Walter Bishop science quotes",
            "expected_results": [
                {"episode_id": 1, "relevance": 2},
                {"episode_id": 8, "relevance": 2},
                {"episode_id": 16, "relevance": 1}
            ]
        },
        {
            "query": "Peter Bishop alternate universe",
            "expected_results": [
                {"episode_id": 45, "relevance": 2},
                {"episode_id": 22, "relevance": 2},
                {"episode_id": 43, "relevance": 1}
            ]
        },
        {
            "query": "shapeshifters episodes",
            "expected_results": [
                {"episode_id": 27, "relevance": 2},
                {"episode_id": 14, "relevance": 1},
                {"episode_id": 35, "relevance": 1}
            ]
        }
    ]

    with open('sample_golden_dataset.json', 'w') as f:
        json.dump(sample_golden, f, indent=2)

    print("✅ Created sample golden dataset")
    return sample_golden

def demo_fringe_ir_system():
    """Demo function to show how to use the system"""
    print("🚀 Fringe IR System Demo")
    print("=" * 50)

    # Create sample golden dataset
    golden_dataset = create_sample_golden_dataset()

    # Initialize system
    ir_system = FringeIRSystem()

    # For demo, we'll assume you have your dataset loaded
    # ir_system.build_system(dataset=your_scraped_dataset)

    # Example queries
    test_queries = [
        "Walter Bishop science experiments",
        "Peter alternate universe",
        "Olivia abilities powers",
        "shapeshifters fringe division"
    ]

    print("\n🔍 Testing search queries...")
    for query in test_queries:
        # results = ir_system.search_all_methods(query)
        # ir_system.display_results(query, results)
        print(f"Query: {query}")

    print("\n📊 Evaluation would be performed here...")
    # evaluation = ir_system.evaluate_system('sample_golden_dataset.json')
    # print("Evaluation Results:", evaluation)

if __name__ == "__main__":
    demo_fringe_ir_system()
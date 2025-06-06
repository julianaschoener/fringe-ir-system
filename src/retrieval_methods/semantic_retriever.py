

import pickle
from src.indexing.inverted_index_builder import InvertedIndex

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False


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

        self.doc_embeddings = self.model.encode(texts)
        self.doc_ids = doc_ids

        dimension = self.doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.doc_embeddings)
        self.faiss_index.add(self.doc_embeddings)

        print(f"✅ Built semantic embeddings for {len(texts)} documents")

    def search(self, query, top_k=10):
        """Perform semantic similarity search"""
        if not SEMANTIC_AVAILABLE or self.doc_embeddings is None:
            print("Error: Semantic search not available or embeddings not built")
            return []

        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        similarities, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx < len(self.doc_ids):
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

            if SEMANTIC_AVAILABLE:
                dimension = self.doc_embeddings.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dimension)
                self.faiss_index.add(self.doc_embeddings)

            print(f"✅ Loaded semantic embeddings from {filename}")

        except FileNotFoundError:
            print(f"Embeddings file {filename} not found. Please build embeddings first.")


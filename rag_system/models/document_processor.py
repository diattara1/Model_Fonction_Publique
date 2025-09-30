import os
import time
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import nltk
from config.settings import settings

class DocumentProcessor:
    def __init__(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model
        self.chunks = []
        self.faiss_index = None
        self.bm25 = None
        self.texts = []
        
        # Download required NLTK data
        nltk.download('punkt_tab', quiet=True)
    
    def load_documents(self, pdf_directory: str):
        """Load and process PDF documents"""
        all_docs = []
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(pdf_directory, filename)
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for p in pages:
                    p.metadata["source"] = file_path
                all_docs.extend(pages)
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        split_docs = splitter.split_documents(all_docs)
        
        # Prepare chunks
        self.chunks = []
        for doc in split_docs:
            meta = doc.metadata or {}
            self.chunks.append({
                "text": doc.page_content,
                "source": meta.get("source", "unknown"),
                "page": int(meta.get("page", 0))
            })
        
        self.texts = [c["text"] for c in self.chunks]
        print(f"Loaded {len(self.chunks)} document chunks")
    
    def build_indices(self):
        """Build FAISS and BM25 indices"""
        print("Building embeddings...")
        embs = self.embed_model.encode(
            self.texts, 
            normalize_embeddings=True, 
            show_progress_bar=True
        )
        
        # FAISS index
        print("Building FAISS index...")
        self.faiss_index = faiss.IndexFlatIP(embs.shape[1])
        self.faiss_index.add(np.array(embs, dtype="float32"))
        
        # BM25 index
        print("Building BM25 index...")
        tokenized_corpus = [word_tokenize(text.lower()) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("Indices built successfully!")
    
    def hybrid_search(self, query: str, top_k: int = 3) -> list:
        """Perform hybrid search using both semantic and keyword matching"""
        start_time = time.time()
        
        # Semantic search
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        search_array = np.array(q_emb, dtype="float32")
        D, I = self.faiss_index.search(search_array, top_k)
        vec_indices = I[0].tolist()
        
        # Keyword search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_bm25 = np.argsort(bm25_scores)[::-1][:top_k].tolist()
        
        # Combine results using Reciprocal Rank Fusion (RRF)
        candidates = set(vec_indices + top_bm25)
        scored = []
        for idx in candidates:
            rank_vec = vec_indices.index(idx) + 1 if idx in vec_indices else 1000
            rank_bm = top_bm25.index(idx) + 1 if idx in top_bm25 else 1000
            rrf = 1/(60+rank_vec) + 1/(60+rank_bm)
            scored.append((rrf, idx))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        keep = [idx for _, idx in scored[:top_k]]
        result = [{**self.chunks[i], "idx": i} for i in keep]
        
        print(f"🔍 Search took {time.time() - start_time:.2f}s")
        return result
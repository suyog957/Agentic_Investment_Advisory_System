"""
ChromaDB-backed knowledge store with sentence-transformers embeddings.

Pipeline: Document → Chunk → Embed → Store → Retrieve
"""

import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import CHROMA_DB_PATH, DOCUMENTS_PATH, EMBEDDING_MODEL


class InvestmentKnowledgeStore:
    """Vector store for investment knowledge documents."""

    CHUNK_SIZE = 400       # words per chunk
    CHUNK_OVERLAP = 50     # overlapping words between chunks

    def __init__(self):
        self._client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self._collection = self._client.get_or_create_collection(
            name="investment_knowledge",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._ensure_documents_loaded()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, query_text: str, n_results: int = 5) -> str:
        """Return the most relevant document chunks for a query."""
        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self._collection.count()),
        )
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        formatted = []
        for chunk, meta in zip(chunks, metadatas):
            source = meta.get("source", "unknown")
            formatted.append(f"[Source: {source}]\n{chunk}")

        return "\n\n---\n\n".join(formatted)

    def document_count(self) -> int:
        return self._collection.count()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_documents_loaded(self) -> None:
        """Load documents if the collection is empty (first run)."""
        if self._collection.count() > 0:
            return  # Already populated

        print("  [KnowledgeStore] First run — loading documents into ChromaDB...")
        for filename in os.listdir(DOCUMENTS_PATH):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(DOCUMENTS_PATH, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            doc_id = filename.replace(".md", "")
            chunks = self._chunk_text(content)
            ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "doc_id": doc_id}] * len(chunks)

            self._collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas,
            )
            print(f"  [KnowledgeStore] Loaded '{filename}' → {len(chunks)} chunks")

        print(f"  [KnowledgeStore] Total chunks: {self._collection.count()}")

    def _chunk_text(self, text: str) -> list[str]:
        """Sliding-window word-based chunking with overlap."""
        words = text.split()
        chunks = []
        step = self.CHUNK_SIZE - self.CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.CHUNK_SIZE])
            if chunk:
                chunks.append(chunk)
        return chunks

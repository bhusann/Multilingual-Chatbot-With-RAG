"""
rag package
===========
RAG system for government scheme documents.
"""

from rag.embeddings import (
    EmbeddingModel,
    embed_texts,
    get_embedding_dim,
    get_embedding_model,
)

from rag.vector_store import (
    VectorStore,
    compute_file_hash,
)

from rag.chunker import chunk_text

from rag.ingest import (
    ingest_document,
    ingest_folder,
)

__all__ = [
    "EmbeddingModel",
    "embed_texts",
    "get_embedding_dim",
    "get_embedding_model",
    "VectorStore",
    "compute_file_hash",
    "chunk_text",
    "ingest_document",
    "ingest_folder",
]

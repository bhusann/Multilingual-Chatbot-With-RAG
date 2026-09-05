"""
vector_store.py
===============
Chroma persistent vector store for the RAG system.

Stores document chunks with embeddings, metadata,
and supports duplicate detection via file hash.

Data lives on disk and survives application restarts.
"""

import os
import hashlib

import chromadb
from chromadb.config import Settings


# Default storage path
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "chroma",
)


class VectorStore:
    """
    Persistent Chroma vector store.

    Each collection stores:
    - chunk text
    - embedding (via Chroma's internal storage)
    - source filename
    - page number
    - section/heading
    - scheme name
    - document ID
    - upload date
    - file hash
    """

    def __init__(self, db_path=DEFAULT_DB_PATH):

        self.db_path = os.path.abspath(db_path)

        os.makedirs(self.db_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
            ),
        )

        # Default collection for government schemes
        self.collection = (
            self.client.get_or_create_collection(
                name="government_schemes",
                metadata={
                    "hnsw:space": "cosine",
                },
            )
        )

        print(
            f"Vector store ready at: {self.db_path}"
        )
        print(
            f"Chunks in store: "
            f"{self.collection.count()}"
        )

    def add_chunks(
        self,
        chunks,
        embeddings,
        doc_id,
        filename,
        file_hash,
        upload_date,
        scheme_name="",
    ):
        """
        Add chunks to the vector store.

        chunks: list of dicts with 'text', 'heading',
                'page' keys
        embeddings: numpy array of shape (n, dim)
        doc_id: unique document identifier
        filename: source filename
        file_hash: SHA-256 hash of the file
        upload_date: ISO format date string
        scheme_name: optional scheme name
        """

        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for i, chunk in enumerate(chunks):

            chunk_id = f"{doc_id}_chunk_{i}"

            ids.append(chunk_id)
            documents.append(chunk["text"])

            metadatas.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_hash": file_hash,
                    "upload_date": upload_date,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "heading": chunk.get(
                        "heading", ""
                    ),
                    "page": chunk.get("page", 0),
                    "scheme_name": scheme_name,
                }
            )

            embeddings_list.append(
                embeddings[i].tolist()
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
        )

        print(
            f"Added {len(chunks)} chunks for "
            f"'{filename}' (doc_id={doc_id})"
        )

    def search(
        self,
        query_embedding,
        n_results=10,
        where=None,
    ):
        """
        Search for similar chunks.

        query_embedding: numpy array of shape (dim,)
        n_results: number of results to return
        optional where: Chroma metadata filter

        Returns list of dicts with 'text', 'metadata',
        'distance' keys.
        """

        kwargs = {
            "query_embeddings": [
                query_embedding.tolist()
            ],
            "n_results": min(
                n_results,
                self.collection.count(),
            ),
        }

        if where:
            kwargs["where"] = where

        # Ensure query_embedding is a flat list
        if isinstance(query_embedding, list):
            # Could be nested: [[...]] -> flatten
            while (
                isinstance(query_embedding, list)
                and len(query_embedding) == 1
                and isinstance(query_embedding[0], list)
            ):
                query_embedding = query_embedding[0]

        kwargs["query_embeddings"] = [
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        ]

        results = self.collection.query(**kwargs)

        formatted = []

        for i in range(
            len(results["ids"][0])
        ):

            formatted.append(
                {
                    "text": results[
                        "documents"
                    ][0][i],
                    "metadata": results[
                        "metadatas"
                    ][0][i],
                    "distance": results[
                        "distances"
                    ][0][i]
                    if results.get("distances")
                    else None,
                }
            )

        return formatted

    def document_exists(self, file_hash):
        """Check if a document with this hash
        already exists."""

        results = self.collection.get(
            where={"file_hash": file_hash},
            limit=1,
        )

        return len(results["ids"]) > 0

    def get_document_ids(self, doc_id):
        """Get all chunk IDs for a document."""

        results = self.collection.get(
            where={"doc_id": doc_id},
        )

        return results["ids"]

    def delete_document(self, doc_id):
        """Delete all chunks for a document."""

        ids = self.get_document_ids(doc_id)

        if ids:
            self.collection.delete(ids=ids)
            print(
                f"Deleted {len(ids)} chunks for "
                f"doc_id={doc_id}"
            )

    def list_documents(self):
        """
        List all unique documents in the store.

        Returns list of dicts with document metadata.
        """

        all_metadata = self.collection.get()[
            "metadatas"
        ]

        seen_docs = {}

        for meta in all_metadata:

            doc_id = meta.get("doc_id", "")

            if doc_id and doc_id not in seen_docs:

                seen_docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get(
                        "filename", ""
                    ),
                    "scheme_name": meta.get(
                        "scheme_name", ""
                    ),
                    "upload_date": meta.get(
                        "upload_date", ""
                    ),
                    "total_chunks": meta.get(
                        "total_chunks", 0
                    ),
                }

        return list(seen_docs.values())

    def count(self):
        """Return total number of chunks."""

        return self.collection.count()


def compute_file_hash(filepath):
    """Compute SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:

        for chunk in iter(
            lambda: f.read(8192), b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()

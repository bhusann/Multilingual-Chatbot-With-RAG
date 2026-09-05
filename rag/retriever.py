"""
retriever.py
============
RAG retrieval module.

Embeds a user query, searches Chroma for relevant
government scheme chunks, and formats them as context
for the LLM.

Flow:
    user question
        -> embed question (Qwen3-Embedding-0.6B)
        -> vector search (Chroma, top 5)
        -> format context block
        -> return (context_text, sources)
"""

from rag.embeddings import embed_texts
from rag.vector_store import VectorStore


# Max chunks to retrieve
TOP_K = 5

# Min similarity (cosine distance). Chunks farther
# than this are discarded as irrelevant.
# Chroma returns cosine distance: 0 = identical, 2 = opposite.
# A threshold of 1.5 means moderately relevant.
MAX_DISTANCE = 1.5


class Retriever:
    """
    Searches the vector store for chunks relevant to
    a user question.
    """

    def __init__(self):
        self.store = VectorStore()

    def search(self, query, top_k=TOP_K):
        """
        Search for chunks relevant to the query.

        Returns list of dicts:
            {
                "text": "...",
                "heading": "...",
                "filename": "...",
                "page": 1,
                "distance": 0.45,
            }
        """

        if self.store.count() == 0:
            return []

        query_embedding = embed_texts(query)

        results = self.store.search(
            query_embedding,
            n_results=min(top_k, self.store.count()),
        )

        # Filter by distance threshold
        filtered = [
            r for r in results
            if r.get("distance") is not None
            and r["distance"] < MAX_DISTANCE
        ]

        return filtered

    def get_context(self, query, top_k=TOP_K):
        """
        Search and format results as a context block
        for the LLM system prompt.

        Returns (context_text, sources_list).
        """

        results = self.search(query, top_k)

        if not results:
            return None, []

        # Build the context block
        lines = []
        sources = []

        for i, r in enumerate(results, 1):
            meta = r.get("metadata", {})
            heading = meta.get("heading", "")
            filename = meta.get("filename", "unknown")
            page = meta.get("page", 0)

            source_tag = filename
            if page:
                source_tag += f" (page {page})"
            if heading:
                source_tag += f" — {heading}"

            lines.append(
                f"[Source {i}: {source_tag}]\n"
                f"{r['text']}"
            )

            sources.append({
                "filename": filename,
                "page": page,
                "heading": heading,
                "text_preview": r["text"][:200],
            })

        context = (
            "RETRIEVED GOVERNMENT SCHEME EVIDENCE:\n"
            "Use the following document chunks to answer "
            "the user's question. If the evidence is "
            "insufficient, say so. Do not invent facts.\n\n"
            + "\n\n".join(lines)
        )

        return context, sources

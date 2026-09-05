"""
ingest.py
=========
Document ingestion pipeline for the RAG system.

Handles:
- PDF text extraction (PyMuPDF)
- Any text-readable file (.txt, .md, .json, .py, .csv, etc.)
- Text chunking
- Embedding generation
- Persistent storage in Chroma

Usage:
    python -m rag.ingest path/to/document.pdf
    python -m rag.ingest path/to/document.md
    python -m rag.ingest path/to/folder/
"""

import os
import sys
import uuid
from datetime import datetime, timezone

# PDF extraction
import pymupdf

from rag.chunker import chunk_text
from rag.embeddings import get_embedding_model
from rag.vector_store import (
    VectorStore,
    compute_file_hash,
)


def extract_text_from_pdf(filepath):
    """
    Extract text from a PDF file.

    Returns list of dicts with 'text' and 'page' keys.
    """

    doc = pymupdf.open(filepath)

    pages = []

    for page_num in range(len(doc)):

        page = doc.load_page(page_num)

        text = page.get_text("text")

        if text and text.strip():

            pages.append(
                {
                    "text": text.strip(),
                    "page": page_num + 1,
                }
            )

    doc.close()

    return pages


def extract_text_from_txt(filepath):
    """
    Read text from any text-readable file.
    Tries UTF-8 first, falls back to latin-1.

    Returns list with a single dict.
    """

    # Try UTF-8 first, then latin-1 (covers almost everything)
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(filepath, "r", encoding=encoding) as f:
                text = f.read()
            if text.strip():
                return [{"text": text.strip(), "page": 0}]
        except (UnicodeDecodeError, ValueError):
            continue

    return []


# All text-readable extensions we support
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".json", ".jsonl", ".jsonc",
    ".csv", ".tsv",
    ".py", ".pyw",
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".kt", ".scala",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp",
    ".go", ".rs", ".swift", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".fish",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".html", ".htm", ".css", ".scss",
    ".sql",
    ".r", ".R", ".lua", ".pl", ".pm",
    ".env", ".gitignore", ".dockerignore",
    ".log", ".diff", ".patch",
    ".tex", ".bib",
    ".srt", ".vtt",
}


def extract_text(filepath):
    """
    Extract text from a file based on extension.

    PDF → pymupdf extraction
    Everything else → read as text (UTF-8/latin-1)

    Returns list of dicts with 'text' and 'page' keys.
    """

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(filepath)

    # Accept any file — try reading as text
    return extract_text_from_txt(filepath)


def ingest_document(
    filepath,
    vector_store=None,
    embedding_model=None,
    scheme_name="",
):
    """
    Ingest a single document into the vector store.

    Steps:
    1. Calculate file hash (duplicate check)
    2. Extract text
    3. Chunk text
    4. Generate embeddings
    5. Store in Chroma

    Returns dict with ingestion results.
    """

    filename = os.path.basename(filepath)

    print(f"\n{'='*50}")
    print(f"Ingesting: {filename}")
    print(f"{'='*50}")

    # ----------------------------------------
    # 1. File hash for duplicate detection
    # ----------------------------------------

    file_hash = compute_file_hash(filepath)

    print(f"File hash: {file_hash[:16]}...")

    # ----------------------------------------
    # 2. Check for duplicates
    # ----------------------------------------

    if vector_store is None:
        vector_store = VectorStore()

    if vector_store.document_exists(file_hash):

        print(
            f"⚠️ Document already exists "
            f"(hash match). Skipping."
        )

        return {
            "status": "skipped",
            "reason": "duplicate",
            "filename": filename,
            "file_hash": file_hash,
        }

    # ----------------------------------------
    # 3. Extract text
    # ----------------------------------------

    print("Extracting text...")

    pages = extract_text(filepath)

    if not pages:

        print(
            "⚠️ No text extracted. "
            "Document may be scanned/image-only."
        )

        return {
            "status": "failed",
            "reason": "no_text",
            "filename": filename,
            "file_hash": file_hash,
        }

    total_chars = sum(
        len(p["text"]) for p in pages
    )

    print(
        f"Extracted {len(pages)} pages, "
        f"{total_chars} characters"
    )

    # ----------------------------------------
    # 4. Chunk text
    # ----------------------------------------

    print("Chunking text...")

    all_chunks = []

    for page_data in pages:

        page_chunks = chunk_text(page_data["text"])

        for chunk in page_chunks:

            chunk["page"] = page_data["page"]

            all_chunks.append(chunk)

    print(f"Created {len(all_chunks)} chunks")

    if not all_chunks:

        print("⚠️ No chunks created.")

        return {
            "status": "failed",
            "reason": "no_chunks",
            "filename": filename,
            "file_hash": file_hash,
        }

    # ----------------------------------------
    # 5. Generate embeddings
    # ----------------------------------------

    print("Generating embeddings...")

    if embedding_model is None:
        embedding_model = get_embedding_model()

    texts_to_embed = [
        c["text"] for c in all_chunks
    ]

    embeddings = embedding_model.embed(texts_to_embed)

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    # ----------------------------------------
    # 6. Store in Chroma
    # ----------------------------------------

    print("Storing in vector database...")

    doc_id = uuid.uuid4().hex[:12]

    upload_date = datetime.now(
        timezone.utc
    ).isoformat()

    vector_store.add_chunks(
        chunks=all_chunks,
        embeddings=embeddings,
        doc_id=doc_id,
        filename=filename,
        file_hash=file_hash,
        upload_date=upload_date,
        scheme_name=scheme_name,
    )

    print(f"✅ Ingestion complete: {filename}")
    print(f"   doc_id: {doc_id}")
    print(f"   chunks: {len(all_chunks)}")
    print(f"   total chunks in store: "
          f"{vector_store.count()}")

    return {
        "status": "success",
        "filename": filename,
        "doc_id": doc_id,
        "file_hash": file_hash,
        "chunks": len(all_chunks),
        "pages": len(pages),
    }


def ingest_folder(
    folder_path,
    vector_store=None,
    embedding_model=None,
):
    """
    Ingest all PDF/TXT files in a folder.
    """

    if vector_store is None:
        vector_store = VectorStore()

    if embedding_model is None:
        embedding_model = get_embedding_model()

    results = []

    for filename in sorted(os.listdir(folder_path)):

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf" or ext in TEXT_EXTENSIONS or not ext:

            filepath = os.path.join(
                folder_path, filename
            )

            result = ingest_document(
                filepath,
                vector_store=vector_store,
                embedding_model=embedding_model,
            )

            results.append(result)

    return results


# ----------------------------------------
# CLI entry point
# ----------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python -m rag.ingest "
            "<file_or_folder>"
        )

        sys.exit(1)

    path = sys.argv[1]

    if os.path.isfile(path):

        result = ingest_document(path)

        print(f"\nResult: {result}")

    elif os.path.isdir(path):

        results = ingest_folder(path)

        print(f"\n{'='*50}")
        print(f"Ingestion Summary")
        print(f"{'='*50}")

        for r in results:

            status = r["status"]
            filename = r["filename"]

            if status == "success":
                print(
                    f"  ✅ {filename}: "
                    f"{r['chunks']} chunks"
                )
            elif status == "skipped":
                print(
                    f"  ⏭️ {filename}: "
                    f"skipped ({r['reason']})"
                )
            else:
                print(
                    f"  ❌ {filename}: "
                    f"failed ({r['reason']})"
                )

    else:

        print(f"Path not found: {path}")

        sys.exit(1)

"""
chunker.py
==========
Text chunking for the RAG system.

Splits documents into meaningful chunks suitable for
embedding and retrieval. Uses heading-aware splitting
when possible to keep related content together.

Target chunk size: ~400-700 tokens (~1600-2800 chars).
Overlap: ~200 chars for context continuity.
"""

import re


# Approximate characters per token (English average)
CHARS_PER_TOKEN = 4

# Target chunk size in characters
TARGET_CHUNK_CHARS = 500 * CHARS_PER_TOKEN  # ~2000 chars

# Overlap in characters
OVERLAP_CHARS = 200

# Minimum chunk size (don't create tiny fragments)
MIN_CHUNK_CHARS = 200


def chunk_text(
    text,
    chunk_size=TARGET_CHUNK_CHARS,
    overlap=OVERLAP_CHARS,
):
    """
    Split text into chunks.

    Strategy:
    1. First try to split on headings/section boundaries.
    2. If sections are too large, split on paragraph breaks.
    3. If still too large, split on sentence boundaries.
    4. As last resort, split on word boundaries.

    Returns list of dicts with 'text' and 'heading' keys.
    """

    if not text or not text.strip():
        return []

    # ----------------------------------------
    # Step 1: Identify section structure
    # ----------------------------------------

    sections = _split_into_sections(text)

    # ----------------------------------------
    # Step 2: Chunk each section
    # ----------------------------------------

    chunks = []

    for heading, section_text in sections:

        if len(section_text) <= chunk_size:

            if len(section_text.strip()) >= MIN_CHUNK_CHARS:
                chunks.append({
                    "text": section_text.strip(),
                    "heading": heading,
                })

        else:

            # Section is too large — split further
            sub_chunks = _split_large_text(
                section_text,
                chunk_size,
                overlap,
            )

            for sc in sub_chunks:

                if len(sc.strip()) >= MIN_CHUNK_CHARS:
                    chunks.append({
                        "text": sc.strip(),
                        "heading": heading,
                    })

    return chunks


def _split_into_sections(text):
    """
    Split text into sections based on common heading patterns.

    Returns list of (heading, body_text) tuples.
    """

    # Common heading patterns in government documents
    heading_pattern = re.compile(
        r"^"
        r"(?:"
        r"(?:\d+[\.\)]\s+)"           # "1. " or "1) "
        r"|(?:[IVX]+[\.\)]\s+)"       # "IV. " or "IV) "
        r"|(?:#+\s+)"                  # "# " markdown
        r"|(?:[A-Z][A-Z\s]{2,}:?\s)"  # "SECTION NAME: " or "SECTION NAME "
        r")"
        r"(.+)$",
        re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No headings found — return entire text
        return [("", text)]

    sections = []

    # Content before first heading
    if matches[0].start() > 0:
        pre_text = text[: matches[0].start()].strip()
        if pre_text:
            sections.append(("", pre_text))

    for i, match in enumerate(matches):

        heading = match.group(0).strip()

        # End of this section is start of next heading (or EOF)
        start = match.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        body = text[start:end].strip()

        if body:
            sections.append((heading, body))

    return sections


def _split_large_text(text, chunk_size, overlap):
    """
    Split a large text block into smaller chunks.

    Tries paragraph → sentence → word boundaries.
    """

    # Try paragraph splits first
    paragraphs = re.split(r"\n\s*\n", text)

    if len(paragraphs) > 1:
        return _merge_small_chunks(
            paragraphs, chunk_size, overlap
        )

    # Try sentence splits
    sentences = re.split(
        r"(?<=[.!?])\s+", text
    )

    if len(sentences) > 1:
        return _merge_small_chunks(
            sentences, chunk_size, overlap
        )

    # Last resort: split on word boundaries
    return _split_by_words(text, chunk_size, overlap)


def _merge_small_chunks(parts, chunk_size, overlap):
    """
    Merge small parts into chunks up to chunk_size.
    """

    chunks = []
    current = ""

    for part in parts:

        if (
            len(current) + len(part) + 1
            <= chunk_size
        ):
            current = (
                (current + " " + part).strip()
                if current
                else part
            )
        else:

            if current:
                chunks.append(current)

            # If single part is too large, split it
            if len(part) > chunk_size:

                sub = _split_by_words(
                    part, chunk_size, overlap
                )
                chunks.extend(sub)
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Add overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, overlap)

    return chunks


def _split_by_words(text, chunk_size, overlap):
    """
    Split text on word boundaries.
    """

    words = text.split()
    chunks = []
    current_words = []
    current_len = 0

    for word in words:

        word_len = len(word) + 1  # +1 for space

        if current_len + word_len <= chunk_size:
            current_words.append(word)
            current_len += word_len
        else:

            if current_words:
                chunks.append(" ".join(current_words))

            current_words = [word]
            current_len = word_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def _add_overlap(chunks, overlap):
    """
    Add overlapping context from previous chunk to
    each subsequent chunk.
    """

    result = [chunks[0]]

    for i in range(1, len(chunks)):

        prev = chunks[i - 1]

        # Take last 'overlap' chars from previous chunk
        overlap_text = prev[-overlap:]

        # Find a clean word boundary
        space_idx = overlap_text.find(" ")

        if space_idx > 0:
            overlap_text = overlap_text[space_idx + 1 :]

        combined = overlap_text + " " + chunks[i]
        result.append(combined.strip())

    return result

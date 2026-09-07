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

    Rules (deliberately strict — a false heading splits a chunk
    and pollutes retrieval, while a missed heading only merges):

    - Numbered ("1. ..."), roman ("IV. ...") and markdown
      ("# ...") headings: must fit on ONE line, <= 120 chars,
      and must not contain URLs or print timestamps.
    - ALLCAPS headings ("ELIGIBILITY CRITERIA:"): the whole
      line must be uppercase-ish and <= 80 chars, AND it must
      either end with a colon or be followed by a blank line.
      This rejects form-field labels ("PAN Card Number : X",
      "PERMANENT ADDRESS (TAMIL NADU)" followed by an address).

    Returns list of (heading, body_text) tuples.
    """

    numbered = r"(?:\d+[\.\)]\s+.+)"
    roman = r"(?:[IVX]+[\.\)]\s+.+)"
    markdown = r"(?:#+\s+.+)"
    allcaps = r"(?:[A-Z][A-Z \t/\-&()',]{1,77}:?)"

    heading_pattern = re.compile(
        r"^(?P<heading>" + "|".join(
            [numbered, roman, markdown, allcaps]
        ) + r")$",
        re.MULTILINE,
    )

    candidates = []

    for match in heading_pattern.finditer(text):
        heading = match.group("heading").strip()

        if len(heading) > 120:
            continue

        # Boilerplate is never a heading
        if _is_boilerplate(heading):
            continue

        # ALLCAPS branch only: demand colon or blank line after
        if re.fullmatch(allcaps, heading):
            after = text[match.end():]
            follows_blank_line = after.startswith(
                ("\n\n", "\r\n\r\n")
            ) or re.match(r"[ \t]*\r?\n[ \t]*\r?\n", after)
            if not heading.endswith(":") and not follows_blank_line:
                continue

        candidates.append((match, heading))

    if not candidates:
        # No headings found — return entire text
        return [("", text)]

    sections = []

    # Content before first heading
    if candidates[0][0].start() > 0:
        pre_text = text[: candidates[0][0].start()].strip()
        if pre_text:
            sections.append(("", pre_text))

    for i, (match, heading) in enumerate(candidates):

        # End of this section is start of next heading (or EOF)
        start = match.end()
        end = (
            candidates[i + 1][0].start()
            if i + 1 < len(candidates)
            else len(text)
        )

        body = text[start:end].strip()

        if body:
            sections.append((heading, body))

    return sections


# Fragments that come from print headers/footers, never headings
_BOILERPLATE_RES = (
    re.compile(r"https?://\S+"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}[,\s]+\d{1,2}:\d{2}"),
    re.compile(r"\bregistration_print\.php\b", re.IGNORECASE),
)


def _is_boilerplate(line):
    """True if the line is print boilerplate (URL, timestamp)."""
    return any(rx.search(line) for rx in _BOILERPLATE_RES)


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

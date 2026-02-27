"""Semantic text chunker with markdown-aware splitting.

Merged from RAG 2.0 (chunk_text → list[Chunk]) and TKB (table-aware
chunking, sanitize_for_graphiti, split_large_content for KG episodes).
"""

from __future__ import annotations

import hashlib
import re

from rag_core.config import get_settings
from rag_core.models import Chunk

# ── KG episode utilities (from TKB) ─────────────────────────────

MAX_EPISODE_CHARS = 8_000

_LUCENE_SPECIAL_RE = re.compile(r'[+\-&|!(){}[\]^"~*?:\\/]')


def sanitize_for_graphiti(text: str) -> str:
    """Remove Lucene special characters that break Neo4j fulltext queries."""
    return _LUCENE_SPECIAL_RE.sub(" ", text)


def split_large_content(
    text: str,
    source: str,
    max_chars: int = MAX_EPISODE_CHARS,
) -> list[tuple[str, str]]:
    """Split large text into episode-sized pieces for Graphiti.

    Returns list of (content, source_name) tuples.
    """
    if len(text) <= max_chars:
        return [(text, source)]

    paragraphs = re.split(r"\n\s*\n", text)
    parts: list[tuple[str, str]] = []
    current = ""
    part_num = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > max_chars:
            if current:
                parts.append((current.strip(), f"{source}_part_{part_num}"))
                part_num += 1
                current = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 > max_chars:
                    if current:
                        parts.append((current.strip(), f"{source}_part_{part_num}"))
                        part_num += 1
                    current = sent
                else:
                    current = f"{current} {sent}".strip() if current else sent
            continue

        if len(current) + len(para) + 2 > max_chars:
            if current:
                parts.append((current.strip(), f"{source}_part_{part_num}"))
                part_num += 1
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        parts.append((current.strip(), f"{source}_part_{part_num}"))

    return parts


# ── Main chunk_text function (from RAG 2.0) ─────────────────────

def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk text semantically using markdown structure.

    Strategy:
    1. Split by markdown headers (##, ###) first
    2. Then by paragraphs
    3. If still too large, split by sentences
    4. Tables (lines starting with |) kept as atomic units

    Each chunk gets auto-generated id (md5) and metadata.
    """
    cfg = get_settings()
    if chunk_size is None:
        chunk_size = cfg.indexing.chunk_size
    if chunk_overlap is None:
        chunk_overlap = cfg.indexing.chunk_overlap

    if not text.strip():
        return []

    chunks: list[Chunk] = []
    sections = _split_by_headers(text)

    for section_title, section_content in sections:
        section_chunks = _chunk_section(section_content, chunk_size, chunk_overlap, section_title)
        chunks.extend(section_chunks)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


# ── Internal helpers ─────────────────────────────────────────────

def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split text by markdown headers (## or ###)."""
    header_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_content: list[str] = []

    for line in text.split("\n"):
        match = header_pattern.match(line)
        if match:
            if current_content:
                sections.append((current_title, "\n".join(current_content)))
            current_title = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append((current_title, "\n".join(current_content)))

    if not sections:
        sections.append(("", text))

    return sections


def _chunk_section(
    text: str, chunk_size: int, chunk_overlap: int, section_title: str,
) -> list[Chunk]:
    """Chunk a single section into Chunk objects."""
    if not text.strip():
        return []

    lines = text.split("\n")
    is_table = all(line.strip().startswith("|") or not line.strip() for line in lines)

    if is_table and text.strip():
        return [_create_chunk(text, section_title)]

    paragraphs = text.split("\n\n")
    chunks: list[Chunk] = []
    current_chunk_text = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk_text) + len(para) + 2 <= chunk_size:
            current_chunk_text = f"{current_chunk_text}\n\n{para}".strip() if current_chunk_text else para
        else:
            if current_chunk_text:
                chunks.append(_create_chunk(current_chunk_text, section_title))
                if chunk_overlap > 0:
                    current_chunk_text = current_chunk_text[-chunk_overlap:] + "\n\n" + para
                else:
                    current_chunk_text = para
            else:
                sentence_chunks = _split_by_sentences(para, chunk_size, chunk_overlap)
                for sc in sentence_chunks:
                    chunks.append(_create_chunk(sc, section_title))
                current_chunk_text = ""

    if current_chunk_text:
        chunks.append(_create_chunk(current_chunk_text, section_title))

    return chunks


def _split_by_sentences(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text by sentence boundaries when paragraph is too large."""
    sentence_pattern = re.compile(r"([.!?]+\s+)")
    parts = sentence_pattern.split(text)

    sentences: list[str] = []
    current = ""
    for i, part in enumerate(parts):
        current += part
        if i % 2 == 1:
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    chunks: list[str] = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) <= chunk_size:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
                if chunk_overlap > 0:
                    current_chunk = current_chunk[-chunk_overlap:] + sent
                else:
                    current_chunk = sent
            else:
                chunks.append(sent)
                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _create_chunk(content: str, section_title: str) -> Chunk:
    """Create Chunk with auto-generated id and metadata."""
    chunk_id = hashlib.md5(content.encode()).hexdigest()[:8]
    return Chunk(
        id=chunk_id,
        content=content,
        metadata={"section_title": section_title} if section_title else {},
    )


# ── Geology-aware chunking ─────────────────────────────────────────

_IMAGE_PATTERN = re.compile(r"!\[|(\[Image)")


def _has_figures(text: str) -> bool:
    """Detect figure references in chunk text."""
    return bool(_IMAGE_PATTERN.search(text))


def _fix_unclosed_code_blocks(text: str) -> str:
    """Append closing ``` if code fences are unbalanced."""
    count = text.count("```")
    if count % 2 != 0:
        text = text.rstrip() + "\n```"
    return text


def _is_table_block(text: str) -> bool:
    """Check if text is predominantly a pipe-format markdown table."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return False
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    return (pipe_lines / len(lines)) > 0.6


def _is_code_block(text: str) -> bool:
    """Check if text is wrapped in code fences."""
    stripped = text.strip()
    return stripped.startswith("```") and stripped.endswith("```")


def _filter_empty_sections(text: str) -> str:
    """Remove empty sections (heading followed immediately by another heading)."""
    lines = text.splitlines()
    result = []
    for i, line in enumerate(lines):
        if line.startswith("#") and i + 1 < len(lines):
            next_non_empty = None
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    next_non_empty = lines[j]
                    break
            if next_non_empty and next_non_empty.startswith("#"):
                continue
        result.append(line)
    return "\n".join(result)


def _load_tokenizer():
    """Try to load the multilingual-e5-large tokenizer.

    Returns (tokenize_to_list, count_tokens) tuple, or None on failure.
    """
    try:
        import warnings

        from transformers import AutoTokenizer

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Token indices sequence length")
            tok = AutoTokenizer.from_pretrained(
                "intfloat/multilingual-e5-large", use_fast=True,
            )
        tok.model_max_length = 1_000_000

        def _encode_list(text: str) -> list[int]:
            return tok.encode(text, add_special_tokens=False)

        def _count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))

        return _encode_list, _count
    except Exception:
        return None


def chunk_geology(
    markdown: str,
    source_file: str = "",
    source_type: str = "",
    source_url: str | None = None,
    software_product: str | None = None,
    language: str = "ru",
    section_path: str = "",
    geology_subdomain: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk markdown with geology-aware metadata and subdomain classification.

    Uses LlamaIndex MarkdownNodeParser + SentenceSplitter for two-stage chunking.
    Applies keyword-based subdomain classification to each chunk.
    Returns list of Chunk objects with geology metadata populated.

    Args:
        markdown: Source markdown text.
        source_file: Original filename.
        source_type: competitor_manual | standard | paper | video | forum.
        source_url: URL if available.
        software_product: Micromine | Surpac | GEOMIX | etc.
        language: "ru" | "en".
        section_path: Heading hierarchy prefix.
        geology_subdomain: Override subdomain (auto-classified if empty/general).
        chunk_size: Target chunk size in tokens (default from config).
        chunk_overlap: Overlap between consecutive chunks (default from config).
    """
    from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
    from llama_index.core.schema import Document

    from rag_core.geology_classifier import classify_subdomain

    cfg = get_settings()
    if chunk_size is None:
        chunk_size = cfg.indexing.chunk_size
    if chunk_overlap is None:
        chunk_overlap = cfg.indexing.chunk_overlap

    if not markdown.strip():
        return []

    # Pre-processing
    text = _fix_unclosed_code_blocks(markdown)
    text = _filter_empty_sections(text)

    # Set up tokenizer
    tok_result = _load_tokenizer()
    if tok_result is not None:
        tok_list_fn, tok_count_fn = tok_result
    else:
        tok_list_fn = lambda t: t.split()  # noqa: E731
        tok_count_fn = lambda t: len(t.split())  # noqa: E731

    # Stage 1: Structure-aware split by headings
    md_parser = MarkdownNodeParser()
    doc = Document(text=text)
    nodes = md_parser.get_nodes_from_documents([doc])

    # Stage 2: Sentence split for oversized nodes
    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        tokenizer=tok_list_fn,
    )

    final_chunks: list[tuple[str, str | None]] = []  # (text, header_path)

    for node in nodes:
        node_text = node.get_content()
        if not node_text.strip():
            continue

        header_path = node.metadata.get("Header_path")
        token_count = tok_count_fn(node_text)

        # Atomic units: tables and code blocks are never split
        if _is_table_block(node_text) or _is_code_block(node_text):
            final_chunks.append((node_text, header_path))
            continue

        # Small enough — keep as-is
        if token_count <= chunk_size:
            final_chunks.append((node_text, header_path))
            continue

        # Oversized — run through SentenceSplitter
        sub_nodes = sentence_splitter.get_nodes_from_documents(
            [Document(text=node_text)]
        )
        for sub in sub_nodes:
            sub_text = sub.get_content()
            if sub_text.strip():
                final_chunks.append((sub_text, header_path))

    # Build Chunk objects with geology metadata
    chunks: list[Chunk] = []

    for idx, (chunk_text, header_path) in enumerate(final_chunks):
        # Build full section path
        parts = []
        if section_path:
            parts.append(section_path)
        elif software_product:
            parts.append(software_product)
        if header_path:
            parts.append(header_path)
        full_section_path = " > ".join(parts) if parts else ""

        # Prepend section path for retrieval context
        if full_section_path:
            context = f"[{full_section_path}]"
        else:
            context = ""

        # Determine subdomain
        subdomain = geology_subdomain
        if not subdomain or subdomain == "general":
            subdomain = classify_subdomain(chunk_text)

        chunk_id = hashlib.md5(
            f"{source_file}::{idx}::{chunk_text[:100]}".encode()
        ).hexdigest()[:12]

        chunk = Chunk(
            id=chunk_id,
            content=chunk_text,
            context=context,
            source_file=source_file,
            source_type=source_type,
            source_url=source_url,
            software_product=software_product,
            geology_subdomain=subdomain,
            section_path=full_section_path,
            language=language,
            has_figures=_has_figures(chunk_text),
            chunk_index=idx,
        )
        chunks.append(chunk)

    return chunks

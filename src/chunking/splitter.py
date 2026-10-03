from typing import Optional
from src.models import Document, DocumentChunk


class RecursiveChunker:
    """
    Recursive Character Chunker (Week 3 concept).
    Splits text hierarchically:
    1. Paragraphs ('\\n\\n')
    2. Single lines ('\\n')
    3. Sentences ('. ', '? ', '! ')
    4. Words (' ')
    5. Raw characters
    With configurable chunk_size and overlap.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " "]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the first separator that works."""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # Fallback: slice by chunk_size
            return [
                text[i : i + self.chunk_size].strip()
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
                if text[i : i + self.chunk_size].strip()
            ]

        sep = separators[0]
        remaining_seps = separators[1:]

        # Split on the current separator
        splits = text.split(sep)
        chunks: list[str] = []
        current_chunk = ""

        for part in splits:
            if not part:
                continue

            # Re-attach separator for natural spacing if it's not the first element
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Apply overlap: keep trailing characters of current chunk
                    if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                        overlap_prefix = current_chunk[-self.chunk_overlap :]
                        candidate = f"{overlap_prefix}{sep}{part}"
                    else:
                        candidate = part

                # If a single part is still larger than chunk_size, recurse with next separator
                if len(candidate) > self.chunk_size:
                    sub_chunks = self._split_text(candidate, remaining_seps)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = candidate

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def split_document(self, doc: Document) -> list[DocumentChunk]:
        """Split a single Document into DocumentChunk objects."""
        raw_chunks = self._split_text(doc.content, self.separators)
        chunks: list[DocumentChunk] = []

        for idx, text in enumerate(raw_chunks):
            if not text.strip():
                continue

            chunk_id = f"{doc.doc_id}_c{idx:03d}" if doc.doc_id else f"chunk_{idx:03d}"
            
            # Merge document metadata with chunk-level metadata
            chunk_metadata = doc.metadata.copy()
            chunk_metadata.update({
                "source": doc.source,
                "title": doc.title,
                "char_count": len(text),
                "word_count": len(text.split()),
            })

            chunk_obj = DocumentChunk(
                content=text,
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                chunk_index=idx,
                metadata=chunk_metadata,
            )
            chunks.append(chunk_obj)

        return chunks

    def split_documents(self, docs: list[Document]) -> list[DocumentChunk]:
        """Split a list of Documents into DocumentChunks."""
        all_chunks: list[DocumentChunk] = []
        for doc in docs:
            all_chunks.extend(self.split_document(doc))
        return all_chunks


class FixedSizeChunker:
    """
    Fixed-size sliding window chunker (character or token level).
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, doc: Document) -> list[DocumentChunk]:
        text = doc.content
        step = self.chunk_size - self.chunk_overlap
        chunks = []
        idx = 0
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{doc.doc_id}_c{idx:03d}" if doc.doc_id else f"chunk_{idx:03d}"
                chunk_meta = doc.metadata.copy()
                chunk_meta.update({
                    "source": doc.source,
                    "title": doc.title,
                    "char_count": len(chunk_text),
                })
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        chunk_index=idx,
                        metadata=chunk_meta,
                    )
                )
                idx += 1

            start += step

        return chunks

    def split_documents(self, docs: list[Document]) -> list[DocumentChunk]:
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.split_document(doc))
        return all_chunks

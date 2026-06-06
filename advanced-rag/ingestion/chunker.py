import logging
from dataclasses import dataclass
from typing import Iterable, List

from config import CHUNK_OVERLAP, CHUNK_SIZE
from ingestion.loader import Document, DocumentChunk

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_text(self, text: str) -> List[str]:
        tokens = text.split()
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk = " ".join(tokens[start:end])
            chunks.append(chunk)
            if end == len(tokens):
                break
            start = max(end - self.overlap, end)
        return chunks

    def chunk_documents(self, documents: Iterable[Document]) -> List[DocumentChunk]:
        chunks = []
        for document in documents:
            text_chunks = self._split_text(document.text)
            for index, chunk in enumerate(text_chunks, start=1):
                chunk_id = f"{document.doc_id}_chunk_{index}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    title=document.title,
                    text=chunk,
                    metadata={"doc_id": document.doc_id, "title": document.title, "chunk_id": chunk_id},
                ))
            logger.debug("Document %s produced %d chunks", document.doc_id, len(text_chunks))

        logger.info("Created %d chunks from documents", len(chunks))
        return chunks

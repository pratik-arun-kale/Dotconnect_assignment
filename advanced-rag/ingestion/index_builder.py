import logging
from typing import List

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME
from ingestion.loader import DocumentChunk

logger = logging.getLogger(__name__)


class IndexBuilder:
    def __init__(self, persist_directory: str = str(CHROMA_DIR)):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        logger.info("Initialized ChromaDB client with persistence %s", self.persist_directory)

    def build_collection(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

        collection = self.client.create_collection(name=COLLECTION_NAME)
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        logger.info("Built ChromaDB collection %s with %d chunks", COLLECTION_NAME, len(chunks))

    def load_collection(self):
        return self.client.get_collection(name=COLLECTION_NAME)

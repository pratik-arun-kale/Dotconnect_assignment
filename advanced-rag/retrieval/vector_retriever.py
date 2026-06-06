import logging
from typing import Dict, List

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(self, collection: Collection, model_name: str = EMBEDDING_MODEL):
        self.collection = collection
        self.embedder = SentenceTransformer(model_name)
        logger.info("Vector retriever loaded embedding model %s", model_name)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        query_embedding = self.embedder.encode(query, convert_to_numpy=True).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )
        if not results or not results["ids"]:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        retrieved = []
        for rank, (chunk_id, distance, text, metadata) in enumerate(
            zip(ids, distances, documents, metadatas), start=1
        ):
            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "score": float(1.0 / (1.0 + distance)),
                    "rank": rank,
                    "text": text,
                    "metadata": metadata,
                }
            )
        logger.debug("Vector retriever returned %d candidates", len(retrieved))
        return retrieved

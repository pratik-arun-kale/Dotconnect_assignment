import logging
import re
from typing import Dict, List

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [token for token in tokens if len(token) > 1]


class BM25Retriever:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.documents = [chunk["text"] for chunk in chunks]
        self.tokenized_documents = [_tokenize(text) for text in self.documents]
        self.model = BM25Okapi(self.tokenized_documents)
        logger.info("BM25 retriever initialized with %d chunks", len(self.chunks))

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        query_tokens = _tokenize(query)
        scores = self.model.get_scores(query_tokens)
        ranked = sorted(
            [
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": float(scores[i]),
                    "rank": rank + 1,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
                for i, chunk in enumerate(self.chunks)
                for rank in [0]
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        top_results = ranked[:top_k]
        for index, item in enumerate(top_results, start=1):
            item["rank"] = index
        logger.debug("BM25 retrieved %d candidates for query: %s", len(top_results), query)
        return top_results

import logging
from typing import Dict, List

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)
        logger.info("Loaded reranker model %s", model_name)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self.model.predict(pairs)

        ranked = []
        for candidate, score in zip(candidates, scores):
            ranked.append(
                {
                    **candidate,
                    "reranker_score": float(score),
                }
            )
        ranked = sorted(ranked, key=lambda item: item["reranker_score"], reverse=True)
        for index, item in enumerate(ranked[:top_k], start=1):
            item["reranked_rank"] = index
        logger.debug("Reranked top %d candidates", min(len(ranked), top_k))
        return ranked[:top_k]

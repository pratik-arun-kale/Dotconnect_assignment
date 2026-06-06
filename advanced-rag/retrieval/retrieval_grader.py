import logging
import re
from typing import Dict, List, Tuple

from config import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

SYNONYM_MAP = {
    "remote": ["telework", "distributed work", "work from home"],
    "approval": ["authorization", "sign-off", "permission"],
    "security": ["safety", "compliance", "risk management"],
    "expense": ["reimbursement", "costs", "spending"],
    "leave": ["vacation", "time off", "absence"],
}


def _expand_tokens(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    expanded = []
    for token in tokens:
        expanded.append(token)
        if token in SYNONYM_MAP:
            expanded.extend(SYNONYM_MAP[token])
    return expanded


def expand_query(query: str) -> str:
    tokens = _expand_tokens(query)
    expanded_query = " ".join(tokens)
    logger.info("Expanded query from [%s] to [%s]", query, expanded_query)
    return expanded_query


def grade_retrieval(reranked_candidates: List[Dict]) -> Tuple[float, bool]:
    if not reranked_candidates:
        return 0.0, True
    average_score = sum(candidate.get("reranker_score", 0.0) for candidate in reranked_candidates) / len(reranked_candidates)
    needs_retrieval = average_score < CONFIDENCE_THRESHOLD
    logger.debug("Average reranker score=%.4f threshold=%.4f needs_correction=%s", average_score, CONFIDENCE_THRESHOLD, needs_retrieval)
    return average_score, needs_retrieval

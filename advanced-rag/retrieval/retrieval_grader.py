import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Score thresholds tuned for cross-encoder/ms-marco-MiniLM-L-6-v2.
# That model scores genuinely relevant passages in the +2 to +5 range and
# irrelevant passages around -10 to -11. Averaging all candidates is therefore
# misleading: a highly relevant top result paired with many irrelevant ones
# produces a strongly negative average even when retrieval succeeded.
# Top score and score margin (gap to the second result) are the right signals —
# they measure both absolute relevance and retrieval certainty independently.
_HIGH_SCORE_MIN = 2.0
_HIGH_MARGIN_MIN = 5.0
_MEDIUM_SCORE_MIN = 0.0
_CORRECTIVE_MARGIN_MIN = 1.0

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


def grade_retrieval(reranked_candidates: List[Dict]) -> Tuple[float, float, str, bool]:
    """
    Returns (top_score, score_margin, confidence_level, needs_corrective_retrieval).

    confidence_level is one of "HIGH", "MEDIUM", or "LOW":
      HIGH   — top result is clearly relevant and far ahead of the pack
      MEDIUM — top result has a positive score but separation is modest
      LOW    — top result is irrelevant or two results are nearly tied (ambiguous)

    needs_corrective_retrieval is True only when retrieval has genuinely failed:
      - no candidates at all
      - top score is negative (cross-encoder says the best match is irrelevant)
      - score margin < 1 (top and second result are nearly indistinguishable)

    HIGH confidence explicitly forces needs_corrective_retrieval=False.
    This is a deliberate safeguard: HIGH already means the top result is
    clearly relevant with a large separation from the pack, so corrective
    retrieval would only replace good results with noisier ones.
    """
    if not reranked_candidates:
        return 0.0, 0.0, "LOW", True

    top_score = reranked_candidates[0].get("reranker_score", 0.0)
    second_score = (
        reranked_candidates[1].get("reranker_score", -999.0)
        if len(reranked_candidates) > 1
        else -999.0
    )
    score_margin = top_score - second_score

    if top_score > _HIGH_SCORE_MIN and score_margin > _HIGH_MARGIN_MIN:
        confidence_level = "HIGH"
    elif top_score > _MEDIUM_SCORE_MIN:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # HIGH confidence → corrective retrieval is never warranted.
    # MEDIUM/LOW → trigger when top result is negative or separation is ambiguous.
    if confidence_level == "HIGH":
        needs_corrective_retrieval = False
    else:
        needs_corrective_retrieval = (
            len(reranked_candidates) == 0
            or top_score < _MEDIUM_SCORE_MIN
            or score_margin < _CORRECTIVE_MARGIN_MIN
        )

    logger.info(
        "Top reranker score: %.4f | Second reranker score: %.4f | Score margin: %.4f | "
        "Confidence level: %s | Corrective retrieval triggered: %s",
        top_score,
        second_score,
        score_margin,
        confidence_level,
        needs_corrective_retrieval,
    )
    return top_score, score_margin, confidence_level, needs_corrective_retrieval

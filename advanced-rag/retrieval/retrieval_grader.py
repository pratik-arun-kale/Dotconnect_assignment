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

_STOP_WORDS = {
    "is", "are", "was", "were", "be", "been", "the", "a", "an",
    "how", "what", "when", "where", "why", "which", "who",
    "does", "do", "did", "for", "to", "of", "in", "at", "by",
    "on", "with", "from", "this", "that",
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


def simplify_query(query: str) -> str:
    """
    Strips stop words from query and returns space-joined remaining tokens
    with length > 2. Used as a corrective retrieval strategy when the initial
    retrieval confidence is low or ambiguous.
    """
    tokens = re.findall(r"\b\w+\b", query.lower())
    simplified_tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    simplified_query = " ".join(simplified_tokens)
    logger.info("Simplified query from [%s] to [%s]", query, simplified_query)
    return simplified_query


def _build_confidence_reason(
    top_score: float,
    score_margin: float,
    confidence_level: str,
    bm25_rank: int,
    vector_rank: int,
) -> str:
    """
    Builds a human-readable single string describing the confidence decision,
    incorporating the reranker scores, retrieval agreement between BM25 and
    vector search, and the resulting confidence level.
    """
    # Describe absolute score strength
    if top_score > _HIGH_SCORE_MIN:
        score_desc = f"Top reranker score ({top_score:.2f}) is strong"
    elif top_score > _MEDIUM_SCORE_MIN:
        score_desc = f"Top reranker score ({top_score:.2f}) is moderate"
    else:
        score_desc = f"Top reranker score ({top_score:.2f}) is weak"

    # Describe margin / separation
    if score_margin > _HIGH_MARGIN_MIN:
        margin_desc = f"substantially ahead of second candidate (margin {score_margin:.2f})"
    elif score_margin >= _CORRECTIVE_MARGIN_MIN:
        margin_desc = f"ahead of second candidate (margin {score_margin:.2f})"
    else:
        margin_desc = (
            f"close to second candidate - retrieval is ambiguous (margin {score_margin:.2f})"
        )

    # Describe retrieval agreement
    bm25_retrieved = bm25_rank <= 10
    vector_retrieved = vector_rank <= 10
    if bm25_retrieved and vector_retrieved:
        retrieval_desc = (
            f"Both BM25 (rank {bm25_rank}) and vector search (rank {vector_rank}) "
            "retrieved this document - strong retrieval agreement."
        )
    elif bm25_retrieved:
        retrieval_desc = f"Document retrieved by BM25 only."
    elif vector_retrieved:
        retrieval_desc = f"Document retrieved by vector search only."
    else:
        retrieval_desc = "Document not retrieved by either method independently."

    # Compose final reason — weak scores skip the margin clause
    if top_score <= _MEDIUM_SCORE_MIN:
        reason = f"{score_desc}. {retrieval_desc}"
    else:
        reason = f"{score_desc}, {margin_desc}. {retrieval_desc}"

    return reason


def grade_retrieval(
    reranked_candidates: List[Dict],
) -> Tuple[float, float, str, bool, str]:
    """
    Returns (top_score, score_margin, confidence_level, needs_corrective_retrieval, reason).

    confidence_level is one of "HIGH", "MEDIUM", or "LOW":
      HIGH   — top result is clearly relevant and far ahead of the pack, or MEDIUM
               upgraded because both BM25 and vector search agree on the top result
      MEDIUM — top result has a positive score but separation is modest, or HIGH
               downgraded because neither BM25 nor vector search retrieved it directly
      LOW    — top result is irrelevant or two results are nearly tied (ambiguous)

    needs_corrective_retrieval is True only when retrieval has genuinely failed:
      - no candidates at all
      - top score is negative (cross-encoder says the best match is irrelevant)
      - score margin < 1 (top and second result are nearly indistinguishable)

    HIGH confidence explicitly forces needs_corrective_retrieval=False.
    This is a deliberate safeguard: HIGH already means the top result is
    clearly relevant with a large separation from the pack, so corrective
    retrieval would only replace good results with noisier ones.

    reason is a human-readable string explaining the confidence decision,
    incorporating reranker scores, retrieval agreement between BM25 and
    vector search, and the resulting confidence level.
    """
    if not reranked_candidates:
        reason = (
            "No candidates returned. Top reranker score (0.00) is weak. "
            "Document not retrieved by either method independently."
        )
        return 0.0, 0.0, "LOW", True, reason

    top_score = reranked_candidates[0].get("reranker_score", 0.0)
    second_score = (
        reranked_candidates[1].get("reranker_score", -999.0)
        if len(reranked_candidates) > 1
        else -999.0
    )
    score_margin = top_score - second_score

    bm25_rank: int = reranked_candidates[0].get("bm25_rank") or 999
    vector_rank: int = reranked_candidates[0].get("vector_rank") or 999

    # Base classification
    if top_score > _HIGH_SCORE_MIN and score_margin > _HIGH_MARGIN_MIN:
        confidence_level = "HIGH"
    elif top_score > _MEDIUM_SCORE_MIN:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    # Composite refinement using retrieval agreement signals
    if (
        confidence_level == "MEDIUM"
        and bm25_rank <= 10
        and vector_rank <= 10
        and top_score > 1.0
    ):
        # Strong retrieval agreement upgrades MEDIUM → HIGH
        confidence_level = "HIGH"
    elif (
        confidence_level == "HIGH"
        and bm25_rank > 10
        and vector_rank > 10
    ):
        # Neither method retrieved it directly — downgrade HIGH → MEDIUM
        confidence_level = "MEDIUM"

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

    reason = _build_confidence_reason(
        top_score, score_margin, confidence_level, bm25_rank, vector_rank
    )

    logger.info(
        "Top reranker score: %.4f | Second: %.4f | Margin: %.4f | "
        "BM25 rank: %s | Vector rank: %s | Confidence: %s | "
        "Corrective: %s | Reason: %s",
        top_score,
        second_score,
        score_margin,
        bm25_rank,
        vector_rank,
        confidence_level,
        needs_corrective_retrieval,
        reason,
    )
    return top_score, score_margin, confidence_level, needs_corrective_retrieval, reason

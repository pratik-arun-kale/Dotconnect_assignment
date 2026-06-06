import re
from typing import List

SPLIT_PATTERNS = [r"\band\b", r"\bor\b", r"\bplus\b", r";", r"&"]
QUESTION_WORDS = ["who", "what", "when", "where", "why", "how", "which", "whom"]


def _normalize_query(query: str) -> str:
    query = query.strip()
    query = re.sub(r"\s+", " ", query)
    query = query.replace("?", " ?")
    return query


def _ends_with_question(text: str) -> bool:
    return text.strip().endswith("?")


def _maybe_add_question_mark(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if not _ends_with_question(text):
        return f"{text}?"
    return text


def transform_query(query: str) -> List[str]:
    """Transform a user query into one or more focused subqueries.

    This function is rule-based and does not use an LLM. It uses punctuation,
    conjunction detection, and simple keyword splitting to decompose compound
    questions into smaller retrieval-friendly queries.
    """
    normalized = _normalize_query(query)
    if "?" in normalized:
        splits = [part.strip() for part in normalized.split("?") if part.strip()]
        return [_maybe_add_question_mark(item) for item in splits]

    lower = normalized.lower()
    if any(conj in lower for conj in [" and ", " or ", " plus ", " & "]):
        tokens = [normalized]
        for pattern in SPLIT_PATTERNS:
            temp = []
            for token in tokens:
                pieces = re.split(pattern, token, flags=re.IGNORECASE)
                if len(pieces) > 1:
                    extended = [piece.strip() for piece in pieces if piece.strip()]
                    temp.extend(extended)
                else:
                    temp.append(token)
            tokens = temp
        if len(tokens) > 1:
            subqueries = [_maybe_add_question_mark(token) for token in tokens if token.strip()]
            return subqueries

    return [_maybe_add_question_mark(query)]

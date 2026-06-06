from __future__ import annotations

import re
from typing import Dict, List

ACRONYM_MAP: Dict[str, str] = {
    "2fa": "two-factor authentication",
    "mfa": "multi-factor authentication",
    "wfh": "work from home",
    "vpn": "virtual private network",
    "sso": "single sign-on",
    "pii": "personally identifiable information",
    "pto": "paid time off",
    "otp": "one-time password",
}

SYNONYM_MAP: Dict[str, str] = {
    "vacation": "leave",
    "holiday": "leave",
    "mandatory": "required",
    "compulsory": "required",
    "obligatory": "required",
    "terminate": "termination",
    "fire": "termination",
    "fired": "termination",
    "resign": "resignation",
    "quit": "resignation",
}

TYPO_MAP: Dict[str, str] = {
    "compulsosry": "compulsory",
    "authentcation": "authentication",
    "authenication": "authentication",
    "authentification": "authentication",
    "reimburment": "reimbursement",
    "reimbusement": "reimbursement",
    "poilcy": "policy",
    "plicy": "policy",
    "polcy": "policy",
    "vacaton": "vacation",
    "securty": "security",
    "seurity": "security",
    "employe": "employee",
    "emploee": "employee",
}

_SPLIT_PATTERN = re.compile(r"\?|(?:\s+(?:and|or|plus|&|;)\s+)", re.IGNORECASE)
_SYNONYM_PATTERNS = {
    word: re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    for word in SYNONYM_MAP
}


def _correct_typos(query: str) -> tuple[str, List[str]]:
    rules: List[str] = []
    tokens = query.split()
    corrected: List[str] = []
    for token in tokens:
        stripped = token.rstrip(".,;:!?")
        suffix = token[len(stripped):]
        lower = stripped.lower()
        if lower in TYPO_MAP:
            replacement = TYPO_MAP[lower]
            if stripped and stripped[0].isupper():
                replacement = replacement.capitalize()
            rules.append(f"typo: '{lower}' -> '{TYPO_MAP[lower]}'")
            corrected.append(replacement + suffix)
        else:
            corrected.append(token)
    return " ".join(corrected), rules


def _expand_acronyms(query: str) -> tuple[str, List[str]]:
    rules: List[str] = []
    tokens = query.split()
    expanded: List[str] = []
    for token in tokens:
        stripped = token.rstrip(".,;:!?")
        suffix = token[len(stripped):]
        lower = stripped.lower()
        if lower in ACRONYM_MAP:
            replacement = ACRONYM_MAP[lower]
            rules.append(f"acronym: '{lower}' -> '{replacement}'")
            expanded.append(replacement + suffix)
        else:
            expanded.append(token)
    return " ".join(expanded), rules


def _normalize_synonyms(query: str) -> tuple[str, List[str]]:
    rules: List[str] = []
    result = query
    for word, pattern in _SYNONYM_PATTERNS.items():
        if pattern.search(result):
            def _replace(m: re.Match, target: str = SYNONYM_MAP[word]) -> str:
                original = m.group(0)
                if original[0].isupper():
                    return target.capitalize()
                return target

            result = pattern.sub(_replace, result)
            rules.append(f"synonym: '{word}' -> '{SYNONYM_MAP[word]}'")
    return result, rules


def _split_query(query: str) -> List[str]:
    parts = _SPLIT_PATTERN.split(query)
    subqueries: List[str] = []
    for part in parts:
        part = part.strip()
        if part:
            subqueries.append(part)
    return subqueries or [query.strip()]


def _ensure_question_marks(subqueries: List[str]) -> List[str]:
    result: List[str] = []
    for sq in subqueries:
        if sq and not sq.endswith("?"):
            sq = sq + "?"
        result.append(sq)
    return result


def transform_query_full(query: str) -> Dict:
    all_rules: List[str] = []

    after_typo, typo_rules = _correct_typos(query)
    all_rules.extend(typo_rules)

    after_acronym, acronym_rules = _expand_acronyms(after_typo)
    all_rules.extend(acronym_rules)

    after_synonym, synonym_rules = _normalize_synonyms(after_acronym)
    all_rules.extend(synonym_rules)

    subqueries = _split_query(after_synonym)
    subqueries = _ensure_question_marks(subqueries)

    if not all_rules:
        all_rules = ["none — query passed through unchanged"]

    return {
        "original_query": query,
        "expanded_query": after_synonym,
        "subqueries": subqueries,
        "rules_applied": all_rules,
    }


def transform_query(query: str) -> List[str]:
    return transform_query_full(query)["subqueries"]

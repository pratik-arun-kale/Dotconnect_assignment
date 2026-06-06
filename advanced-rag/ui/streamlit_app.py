import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from config import (
    EMBEDDING_MODEL,
    FUSED_TOP_K,
    MAX_RETRIEVAL_ATTEMPTS,
    QA_MODEL,
    RERANK_TOP_K,
    RERANKER_MODEL,
    VECTOR_TOP_K,
)
from generation.citation_builder import build_citations
from generation.extractive_qa import ExtractiveQA
from ingestion.build_index import main as build_index
from ingestion.chunker import Chunker
from ingestion.loader import DocumentLoader
from ingestion.index_builder import IndexBuilder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.query_transform import transform_query_full
from retrieval.reranker import Reranker
from retrieval.retrieval_grader import expand_query, grade_retrieval, simplify_query
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.vector_retriever import VectorRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Corrective retrieval strategies tried in order; first that improves top_score wins.
_CORRECTIVE_STRATEGIES = [
    ("synonym_expansion", expand_query),
    ("query_simplification", simplify_query),
]


@st.cache_resource
def load_pipeline() -> Dict:
    loader = DocumentLoader()
    documents = loader.load_documents()
    if not documents:
        raise RuntimeError("No documents were found in the data directory.")

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)
    builder = IndexBuilder()
    collection = None
    try:
        collection = builder.load_collection()
    except Exception:
        collection = None

    if collection is None:
        build_index()
        collection = builder.load_collection()

    bm25_retriever = BM25Retriever([
        {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
        for c in chunks
    ])
    vector_retriever = VectorRetriever(collection, EMBEDDING_MODEL)
    reranker = Reranker(RERANKER_MODEL)
    qa_engine = ExtractiveQA(QA_MODEL)

    return {
        "documents": documents,
        "chunks": chunks,
        "bm25": bm25_retriever,
        "vector": vector_retriever,
        "reranker": reranker,
        "qa": qa_engine,
    }


def _explain_pipeline(response: Dict, original_query: str) -> Dict[str, str]:
    reranked = response["reranked_results"]
    bm25_results = response["bm25_results"]
    vector_results = response["vector_results"]

    if not reranked:
        empty = "No results available for this stage."
        return {k: empty for k in ("bm25", "vector", "rrf", "reranker", "qa")}

    top = reranked[0]
    doc_id = top["metadata"].get("doc_id", "unknown")
    bm25_rank = top.get("bm25_rank")
    vector_rank = top.get("vector_rank")
    rrf_score = top.get("rrf_score", 0)
    reranker_score = top.get("reranker_score", 0)

    # BM25
    if bm25_rank and bm25_rank <= 5:
        bm25_score = next(
            (r["score"] for r in bm25_results if r["metadata"].get("doc_id") == doc_id), 0.0
        )
        bm25_why = (
            f"BM25 Okapi ranked **{doc_id}** at position **{bm25_rank}** (score {bm25_score:.3f}).\n\n"
            f"BM25 assigns scores via TF×IDF weighting with document-length normalization. "
            f"This document had high term overlap with the query — the query keywords appear frequently "
            f"here while being rare across the corpus (high IDF). "
            f"BM25 is strongest for exact keyword matches and proper nouns."
        )
    elif bm25_rank:
        bm25_why = (
            f"BM25 ranked **{doc_id}** at position **{bm25_rank}** — partial keyword overlap only.\n\n"
            f"Query terms appear in this document but may be common across the corpus (low IDF) "
            f"or occur less frequently here (low TF). BM25 can miss synonymous or paraphrased content."
        )
    else:
        bm25_why = (
            f"BM25 did **not** retrieve **{doc_id}** in the top-{VECTOR_TOP_K} results.\n\n"
            f"The document lacked sufficient keyword overlap with the query. "
            f"It was surfaced by vector search and promoted by the cross-encoder — "
            f"a demonstration of why hybrid retrieval is necessary."
        )

    # Vector
    if vector_rank and vector_rank <= 5:
        vec_score = next(
            (r["score"] for r in vector_results if r["metadata"].get("doc_id") == doc_id), 0.0
        )
        vec_why = (
            f"Vector search ranked **{doc_id}** at position **{vector_rank}** "
            f"(cosine similarity {vec_score:.3f}).\n\n"
            f"The sentence embedding model (all-MiniLM-L6-v2) encodes text as 384-dimensional dense vectors. "
            f"This document's embedding was closest to the query vector in the ChromaDB index. "
            f"Dense retrieval captures semantic meaning, so synonymous terms like 'MFA' and "
            f"'multi-factor authentication' produce similar embeddings even without exact keyword overlap."
        )
    elif vector_rank:
        vec_why = (
            f"Vector search ranked **{doc_id}** at position **{vector_rank}**.\n\n"
            f"The document embedding was moderately similar to the query vector but outranked by "
            f"semantically closer documents. Dense retrieval can over-retrieve thematically related "
            f"but off-topic passages — a known failure mode addressed by the cross-encoder reranker."
        )
    else:
        vec_why = (
            f"Vector search did **not** retrieve **{doc_id}** in the top-{VECTOR_TOP_K} results.\n\n"
            f"The document's embedding was not sufficiently close to the query vector in semantic space. "
            f"It was retrieved by BM25 via keyword matching — another demonstration of hybrid retrieval value."
        )

    # RRF
    bm25_str = str(bm25_rank) if bm25_rank else "not retrieved"
    vec_str = str(vector_rank) if vector_rank else "not retrieved"
    rrf_why = (
        f"RRF ranked **{doc_id}** at fusion position **{top.get('fused_rank', '?')}** "
        f"(RRF score {rrf_score:.5f}).\n\n"
        f"Reciprocal Rank Fusion (k=60) computes: `score += 1 / (60 + rank)` for each retrieval system. "
        f"For this document: BM25 rank={bm25_str}, vector rank={vec_str}.\n\n"
        f"Key insight: RRF is **scale-invariant** — it uses only rank positions, not raw scores. "
        f"This avoids the problem of combining scores from heterogeneous systems (BM25 scores in TF-IDF "
        f"units vs cosine similarity). A document appearing in both lists receives additive boosts, "
        f"rewarding retrieval agreement."
    )

    # Reranker
    second = reranked[1] if len(reranked) > 1 else None
    if second:
        second_doc = second["metadata"].get("doc_id", "?")
        second_score_val = second.get("reranker_score", 0)
        margin = reranker_score - second_score_val
        reranker_why = (
            f"Cross-encoder selected **{doc_id}** as rank 1 "
            f"(score {reranker_score:.4f} vs **{second_doc}** at {second_score_val:.4f}, "
            f"margin {margin:.4f}).\n\n"
            f"Unlike bi-encoders that embed query and document independently, cross-encoders "
            f"jointly encode the [query, passage] pair. Full attention allows every query token "
            f"to interact with every document token — yielding much higher accuracy for ranking.\n\n"
            f"The large score margin ({margin:.2f}) indicates the model is highly confident "
            f"that **{doc_id}** is more relevant. A small margin (<1) would signal ambiguity."
        )
    else:
        reranker_why = (
            f"Cross-encoder scored **{doc_id}** at {reranker_score:.4f} (sole candidate).\n\n"
            f"Cross-encoders jointly encode [query, passage] for full attention-based interaction, "
            f"giving much higher precision than bi-encoders at the cost of O(n) inference calls."
        )

    # QA
    ans_text = response["answer"].get("answer", "")
    ans_score = response["answer"].get("score", 0)
    if ans_text and "Insufficient evidence" not in ans_text:
        if len(ans_text) > 150:
            qa_why = (
                f"The extractive model returned the **full top chunk as a passage** "
                f"(fallback path, score {ans_score:.4f}).\n\n"
                f"RoBERTa-base (fine-tuned on SQuAD2) predicts start/end token positions. "
                f"When no high-confidence span is found but retrieval confidence is HIGH/MEDIUM, "
                f"the system returns the full context chunk rather than silencing a known-good document. "
                f"This prevents false negatives when the answer is implicit or spans multiple sentences."
            )
        else:
            qa_why = (
                f"Extracted span: **\"{ans_text}\"** (span score {ans_score:.4f}).\n\n"
                f"RoBERTa-base reads the full [question, context] pair and computes start/end logits "
                f"over all 512 token positions. The span with the highest combined logit is returned. "
                f"Score = sigmoid(best_span_logit / 2) — this avoids softmax dilution over 512 positions "
                f"where even confident predictions yield small per-position probabilities."
            )
    else:
        qa_why = (
            f"No confident span was found and retrieval confidence was LOW.\n\n"
            f"This occurs when: (1) the answer requires synthesizing multiple sentences, "
            f"(2) the question phrasing differs substantially from the document, "
            f"(3) the answer is implicit rather than stated verbatim, or "
            f"(4) the retrieved documents genuinely lack the answer."
        )

    return {
        "bm25": bm25_why,
        "vector": vec_why,
        "rrf": rrf_why,
        "reranker": reranker_why,
        "qa": qa_why,
    }


def run_single_query(query: str, pipeline: Dict) -> Dict:
    bm25 = pipeline["bm25"]
    vector = pipeline["vector"]
    reranker = pipeline["reranker"]
    qa = pipeline["qa"]
    docs = pipeline["documents"]
    chunks = pipeline["chunks"]

    # --- Query transformation (timed once, shared across subqueries) ---
    t0 = time.perf_counter()
    transform_result = transform_query_full(query)
    transform_ms = (time.perf_counter() - t0) * 1000
    transformed_queries = transform_result["subqueries"]

    logger.info(
        "Query transform: original=%r expanded=%r rules=%s subqueries=%d",
        query,
        transform_result["expanded_query"],
        transform_result["rules_applied"],
        len(transformed_queries),
    )

    responses = []
    for idx, subquery in enumerate(transformed_queries):
        logger.info("=== PIPELINE: subquery %d/%d = %r ===", idx + 1, len(transformed_queries), subquery)

        # --- BM25 retrieval ---
        t_bm25 = time.perf_counter()
        bm25_results = bm25.retrieve(subquery, top_k=VECTOR_TOP_K)
        bm25_ms = (time.perf_counter() - t_bm25) * 1000

        # --- Vector retrieval ---
        t_vec = time.perf_counter()
        vector_results = vector.retrieve(subquery, top_k=VECTOR_TOP_K)
        vector_ms = (time.perf_counter() - t_vec) * 1000

        # --- RRF fusion ---
        t_rrf = time.perf_counter()
        fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=FUSED_TOP_K)
        rrf_ms = (time.perf_counter() - t_rrf) * 1000

        # --- Reranking ---
        t_rerank = time.perf_counter()
        reranked = reranker.rerank(subquery, fused, top_k=RERANK_TOP_K)
        rerank_ms = (time.perf_counter() - t_rerank) * 1000

        # --- Confidence grading ---
        top_score, score_margin, confidence_level, needs_correction, confidence_reason = grade_retrieval(reranked)
        attempt = 1
        query_used = subquery
        corrective_triggered = False
        corrective_reason = ""
        corrective_strategy = "none"
        corrective_ms = 0.0

        # --- Corrective retrieval ---
        # Tries up to MAX_RETRIEVAL_ATTEMPTS-1 strategies; adopts first that improves top_score.
        if needs_correction and confidence_level != "HIGH":
            corrective_triggered = True
            corrective_reason = f"Confidence={confidence_level}: {confidence_reason}"
            t_corr = time.perf_counter()

            for strategy_name, query_fn in _CORRECTIVE_STRATEGIES[:MAX_RETRIEVAL_ATTEMPTS - 1]:
                attempt += 1
                corrective_query = query_fn(subquery)
                if corrective_query.strip() == subquery.strip():
                    logger.info("Corrective strategy '%s' produced no change, skipping.", strategy_name)
                    continue

                logger.info("Trying corrective strategy '%s': query=%r", strategy_name, corrective_query)
                c_bm25 = bm25.retrieve(corrective_query, top_k=VECTOR_TOP_K)
                c_vector = vector.retrieve(corrective_query, top_k=VECTOR_TOP_K)
                c_fused = reciprocal_rank_fusion(c_bm25, c_vector, top_k=FUSED_TOP_K)
                c_reranked = reranker.rerank(corrective_query, c_fused, top_k=RERANK_TOP_K)
                c_top, c_margin, c_level, _, c_reason = grade_retrieval(c_reranked)

                if c_top > top_score:
                    reranked = c_reranked
                    bm25_results = c_bm25
                    vector_results = c_vector
                    fused = c_fused
                    top_score = c_top
                    score_margin = c_margin
                    confidence_level = c_level
                    confidence_reason = c_reason
                    query_used = corrective_query
                    corrective_strategy = strategy_name
                    logger.info(
                        "Corrective strategy '%s' improved top_score to %.4f (level=%s)",
                        strategy_name, c_top, c_level,
                    )
                    break
                else:
                    logger.info(
                        "Corrective strategy '%s' did not improve (%.4f <= %.4f)",
                        strategy_name, c_top, top_score,
                    )

            corrective_ms = (time.perf_counter() - t_corr) * 1000

        second_score = top_score - score_margin

        # --- Answer extraction ---
        t_qa = time.perf_counter()
        if confidence_level == "LOW":
            answer = {
                "answer": "Insufficient evidence found in retrieved documents.",
                "score": 0.0,
                "source_doc": None,
                "chunk_id": None,
                "context": None,
            }
        else:
            answer = qa.answer_question(
                subquery, reranked, fallback_to_context=(confidence_level in ("HIGH", "MEDIUM"))
            )
        qa_ms = (time.perf_counter() - t_qa) * 1000

        # Transform latency only charged to the first subquery
        subquery_transform_ms = transform_ms if idx == 0 else 0.0
        total_ms = subquery_transform_ms + bm25_ms + vector_ms + rrf_ms + rerank_ms + corrective_ms + qa_ms

        citations = build_citations([answer])

        response_dict = {
            "subquery": subquery,
            "query_used": query_used,
            "bm25_results": bm25_results,
            "vector_results": vector_results,
            "fused_results": fused,
            "reranked_results": reranked,
            "top_score": top_score,
            "second_score": second_score,
            "score_margin": score_margin,
            "confidence_level": confidence_level,
            "confidence_reason": confidence_reason,
            "corrective_triggered": corrective_triggered,
            "corrective_reason": corrective_reason,
            "corrective_strategy": corrective_strategy,
            "attempt_count": attempt,
            "answer": answer,
            "citations": citations,
            "latency_ms": {
                "transform": subquery_transform_ms,
                "bm25": bm25_ms,
                "vector": vector_ms,
                "rrf": rrf_ms,
                "rerank": rerank_ms,
                "corrective": corrective_ms,
                "qa": qa_ms,
                "total": total_ms,
            },
            "diagnostics": {
                "docs_indexed": len(docs),
                "chunks_indexed": len(chunks),
                "bm25_candidates": len(bm25_results),
                "vector_candidates": len(vector_results),
                "fused_candidates": len(fused),
                "reranked_candidates": len(reranked),
                "chunks_used_for_answer": 1 if answer.get("source_doc") else 0,
            },
        }
        response_dict["interview_explanation"] = _explain_pipeline(response_dict, query)
        responses.append(response_dict)

    return {
        "original_query": query,
        "transform": transform_result,
        "transformed_queries": transformed_queries,
        "responses": responses,
    }


# ─── Render helpers ───────────────────────────────────────────────────────────

def _render_answer_tab(response: Dict) -> None:
    level = response["confidence_level"]
    level_color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(level, "⚪")

    st.markdown(f"**Confidence:** {level_color} {level}")
    st.caption(response["confidence_reason"])

    st.markdown(f"**Query used for retrieval:** `{response['query_used']}`")

    corrective_icon = "✅" if response["corrective_triggered"] else "⬜"
    st.markdown(
        f"**Corrective retrieval:** {corrective_icon} "
        f"{'Triggered' if response['corrective_triggered'] else 'Not triggered'} "
        f"— strategy: `{response['corrective_strategy']}` "
        f"(attempt {response['attempt_count']})"
    )
    if response["corrective_reason"]:
        st.caption(f"Reason: {response['corrective_reason']}")

    st.divider()
    st.markdown("**Answer**")
    st.write(response["answer"]["answer"])
    if response["citations"]:
        st.markdown("**Source**")
        for c in response["citations"]:
            st.write(f"📄 {c}")


def _render_retrieval_tab(response: Dict) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**BM25 Top Results**")
        for item in response["bm25_results"]:
            st.write(
                f"{item['rank']}. `{item['metadata']['doc_id']}` | "
                f"`{item['chunk_id']}` | score: {item['score']:.3f}"
            )

        st.markdown("**RRF Fused Results**")
        for item in response["fused_results"]:
            bm25_r = str(item['bm25_rank']) if item['bm25_rank'] else "–"
            vec_r = str(item['vector_rank']) if item['vector_rank'] else "–"
            st.write(
                f"{item['fused_rank']}. `{item['metadata']['doc_id']}` | "
                f"BM25={bm25_r} vec={vec_r} rrf={item['rrf_score']:.4f}"
            )

    with col2:
        st.markdown("**Vector Top Results**")
        for item in response["vector_results"]:
            st.write(
                f"{item['rank']}. `{item['metadata']['doc_id']}` | "
                f"`{item['chunk_id']}` | score: {item['score']:.3f}"
            )

        st.markdown("**Reranked Results**")
        for item in response["reranked_results"]:
            st.write(
                f"{item['reranked_rank']}. `{item['metadata']['doc_id']}` | "
                f"`{item['chunk_id']}` | score: {item['reranker_score']:.4f}"
            )

    st.divider()
    st.markdown("**Reranker Score Breakdown**")
    st.write(f"Top score: `{response['top_score']:.4f}`")
    st.write(f"Second score: `{response['second_score']:.4f}`")
    st.write(f"Score margin: `{response['score_margin']:.4f}`")


def _render_latency_tab(response: Dict) -> None:
    lat = response["latency_ms"]
    diag = response["diagnostics"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Latency Breakdown**")
        rows = [
            ("Query Transform", lat["transform"]),
            ("BM25 Retrieval", lat["bm25"]),
            ("Vector Retrieval", lat["vector"]),
            ("RRF Fusion", lat["rrf"]),
            ("Reranking", lat["rerank"]),
            ("Corrective Retrieval", lat["corrective"]),
            ("QA Extraction", lat["qa"]),
        ]
        for label, ms in rows:
            bar_len = min(int(ms / 5), 40)
            bar = "█" * bar_len if bar_len > 0 else "▏"
            st.write(f"`{label:<22}` {ms:>7.1f} ms  {bar}")
        st.write(f"**Total: {lat['total']:.1f} ms**")

    with col2:
        st.markdown("**Pipeline Diagnostics**")
        st.write(f"Documents indexed:   `{diag['docs_indexed']}`")
        st.write(f"Chunks indexed:      `{diag['chunks_indexed']}`")
        st.write(f"BM25 candidates:     `{diag['bm25_candidates']}`")
        st.write(f"Vector candidates:   `{diag['vector_candidates']}`")
        st.write(f"After RRF fusion:    `{diag['fused_candidates']}`")
        st.write(f"After reranking:     `{diag['reranked_candidates']}`")
        st.write(f"Chunks used for answer: `{diag['chunks_used_for_answer']}`")

        st.markdown("**Latency Notes**")
        rerank_pct = 100 * lat["rerank"] / lat["total"] if lat["total"] > 0 else 0
        qa_pct = 100 * lat["qa"] / lat["total"] if lat["total"] > 0 else 0
        st.caption(
            f"Reranking accounts for {rerank_pct:.0f}% of total latency — "
            f"this is expected. Cross-encoders are the accuracy bottleneck in RAG pipelines."
        )
        st.caption(
            f"In production: reranking is parallelizable per-candidate, "
            f"and can be replaced with a distilled bi-encoder at ~10% of this latency."
        )


def _render_interview_tab(response: Dict) -> None:
    expl = response.get("interview_explanation", {})
    stages = [
        ("1. BM25 Retrieval", "bm25"),
        ("2. Vector Retrieval", "vector"),
        ("3. RRF Fusion", "rrf"),
        ("4. Cross-Encoder Reranking", "reranker"),
        ("5. Answer Extraction", "qa"),
    ]
    for label, key in stages:
        with st.expander(label):
            st.markdown(expl.get(key, "Explanation not available."))


def _render_failure_examples() -> None:
    st.divider()
    with st.expander("📚 System Examples: When Each Retrieval Method Wins and Fails"):
        st.markdown(
            """
### A — BM25 Succeeds, Vector Struggles

**Query:** `"expense reimbursement form EXP-2024-REV3"`

BM25 matches the exact form code (`EXP-2024-REV3`) because it treats each token as a lookup
key — the code appears verbatim in the expense document, producing a high TF-IDF score.

Vector search **struggles** because dense embeddings compress text into a fixed-size vector.
A specific alphanumeric identifier (`EXP-2024-REV3`) gets averaged into the embedding and loses
its distinguishing signal. The retrieved document may be semantically close to "expense" topics
but fail to surface the document containing the exact code.

**Lesson:** BM25 is essential for queries containing codes, identifiers, proper nouns, or
technical terms that have no semantic paraphrases.

---

### B — Vector Succeeds, BM25 Struggles

**Query:** `"time away from the office regulations"`

BM25 tokenizes this as `["time", "away", "office", "regulations"]`. The leave policy document
uses the term "leave" rather than "time away", so BM25 finds **low term overlap** and ranks
the document poorly.

Vector search **succeeds** because the embedding of "time away from the office" is semantically
close to the embedding of "employee leave policy" — both describe absence from work, and the
model has learned this equivalence from large training corpora.

**Lesson:** Vector retrieval handles synonymy, paraphrase, and conceptual similarity that
exact-match methods miss. This is why dense retrieval dramatically outperforms BM25 on
natural language questions with varied vocabulary.

---

### C — Hybrid Succeeds Where Either Alone Fails

**Query:** `"remote work approval and expense reimbursement policy"`

Splitting on "and" gives two subqueries: **"remote work approval"** and **"expense reimbursement policy"**.

- BM25 on subquery 1: strong on "remote" and "approval" → surfaces `remote_work_policy`
- BM25 on subquery 2: strong on "expense" and "reimbursement" → surfaces `expense_reimbursement`
- Vector on subquery 1: captures the semantic meaning of remote work conditions
- Vector on subquery 2: captures expense-related content

After RRF fusion: both documents are promoted because each appeared in **both** retrieval lists.
Neither single-modality retrieval would surface both documents with high confidence for the
combined query — but the hybrid approach correctly retrieves and ranks both.

**Lesson:** Multi-topic queries benefit from query decomposition + hybrid retrieval.
RRF's additive scoring rewards documents that appear in both retrieval lists,
making the fused ranking more robust than either alone.
            """
        )


def render_results(output: Dict) -> None:
    transform = output.get("transform", {})

    # ── Query transformation summary ──────────────────────────────────────────
    st.subheader("Query Analysis")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(f"**Original query:** {transform.get('original_query', output['original_query'])}")
        expanded = transform.get("expanded_query", "")
        if expanded and expanded.rstrip("?") != transform.get("original_query", "").rstrip("?"):
            st.markdown(f"**Expanded query:** {expanded}")
        subqueries = transform.get("subqueries", [])
        if len(subqueries) > 1:
            st.markdown(f"**Subqueries ({len(subqueries)}):**")
            for sq in subqueries:
                st.write(f"  • {sq}")
    with col_b:
        st.markdown("**Transformations applied:**")
        for rule in transform.get("rules_applied", []):
            st.write(f"  • {rule}")

    st.divider()

    # ── Per-subquery results ───────────────────────────────────────────────────
    for response in output["responses"]:
        header = f"Subquery: {response['subquery']}"
        with st.expander(header, expanded=True):
            tab_answer, tab_retrieval, tab_latency, tab_interview = st.tabs([
                "Answer & Confidence",
                "Retrieval Details",
                "Latency & Diagnostics",
                "Interview Explanation",
            ])
            with tab_answer:
                _render_answer_tab(response)
            with tab_retrieval:
                _render_retrieval_tab(response)
            with tab_latency:
                _render_latency_tab(response)
            with tab_interview:
                _render_interview_tab(response)

    _render_failure_examples()


def main() -> None:
    st.set_page_config(page_title="Advanced RAG System", layout="wide")
    st.title("Advanced Retrieval-Augmented Generation (RAG) Demo")
    st.markdown(
        "Hybrid BM25 + vector retrieval · RRF fusion · cross-encoder reranking · "
        "extractive QA · composite confidence grading · corrective retrieval"
    )

    query = st.text_input("Enter a question for the document collection:")
    if not query:
        st.info("Type a question above to begin. Try: *'is MFA compulsory?'* or *'leave policy'*")
        return

    try:
        pipeline = load_pipeline()
        output = run_single_query(query, pipeline)
        render_results(output)
    except Exception as exc:
        logger.exception("Streamlit application error")
        st.error(f"Unable to complete retrieval: {exc}")


if __name__ == "__main__":
    main()

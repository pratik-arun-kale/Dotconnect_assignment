# Interview Guide — Advanced RAG System

## 1. Complete request flow

1. User enters a natural-language question in the Streamlit UI.
2. **Query transformation** expands acronyms, corrects typos, normalises synonyms, and decomposes compound questions into focused subqueries.
3. **BM25 retrieval** and **vector retrieval** run on each subquery.
4. Results are merged via **Reciprocal Rank Fusion (RRF)**.
5. A **cross-encoder reranker** scores the top-k fused candidates.
6. **Composite confidence grading** evaluates retrieval quality using reranker scores, score margin, and BM25/vector retrieval agreement.
7. If confidence is LOW or MEDIUM with ambiguous separation, **corrective retrieval** tries synonym expansion then query simplification.
8. **Extractive QA** (roberta-base-squad2) extracts a grounded answer span. When confidence is HIGH/MEDIUM and span extraction fails, the top-ranked chunk is returned as a passage fallback.
9. The UI renders the answer, confidence level with reason, per-stage latency, pipeline diagnostics, and interview explanations.

---

## 2. Why confidence scoring matters

Confidence scoring is what separates a robust RAG system from a naive retrieve-and-generate pipeline.

**Without confidence scoring:**
- The system answers every query regardless of retrieval quality.
- Low-quality retrieved documents produce incorrect or hallucinated answers.
- There is no mechanism to detect retrieval failure at runtime.

**With composite confidence scoring:**
- The system can distinguish `(top_score=3.98, margin=14.19)` — clearly retrieved the right document — from `(top_score=1.20, margin=0.80)` — retrieval is ambiguous.
- Retrieval agreement (did both BM25 and vector find this document?) provides a second, independent signal that increases confidence in multi-signal agreement cases.
- The system can gate QA execution: only run the extractive model when evidence quality justifies it.

**Why averaging reranker scores is wrong:**
Cross-encoder scores are not absolute probabilities. A highly relevant document might score +3.98 while irrelevant documents score -10 to -11. Averaging these produces a strongly negative confidence number even when retrieval was perfect. The correct signal is the top score (absolute relevance) and the score margin (relative certainty).

---

## 3. Why corrective retrieval exists

Single-attempt retrieval is brittle. Corrective retrieval is the system's mechanism to recover gracefully:

**When does it trigger?**
- `confidence_level == LOW`: the cross-encoder judges the best available document as irrelevant.
- `score_margin < 1.0`: the top result is nearly tied with the second result — retrieval is ambiguous.

**What strategies does it try?**
1. **Synonym expansion:** appends synonyms/related terms (`leave → vacation time off absence`). Broadens recall for queries with unusual vocabulary.
2. **Query simplification:** strips stop words to focus on key nouns. Prevents stop words from diluting BM25 term matching.

**Why not always use the expanded query?**
Expansion can hurt precision. Adding many synonyms can promote tangentially related documents. The system only adopts corrective results when they strictly improve the top reranker score.

---

## 4. Latency vs accuracy tradeoffs

| Component | Latency | Accuracy impact |
|---|---|---|
| BM25 retrieval | ~2–10 ms | High for keyword queries |
| Vector retrieval | ~5–20 ms | High for semantic queries |
| RRF fusion | <1 ms | Improves recall, no accuracy cost |
| Cross-encoder reranking | ~30–200 ms | Largest accuracy gain per component |
| Extractive QA | ~20–100 ms | Grounded answers, no hallucination |

**The reranker is the latency bottleneck.** For production:
- Replace cross-encoder with a distilled bi-encoder for ~10× speedup at ~5% accuracy cost.
- Cache reranker results for repeated queries.
- Use asynchronous reranking for non-latency-critical paths.
- Limit the reranker candidate pool (RERANK_TOP_K) to control latency linearly.

---

## 5. Retrieval vs reranking tradeoffs

Retrieval (BM25 + vector) optimises for **recall** — it needs to surface the right document somewhere in the top-k results. Reranking optimises for **precision** — it identifies which of those candidates is actually most relevant.

This two-stage design is a deliberate engineering choice:
- Running a cross-encoder over all documents would be prohibitively expensive (O(n) cross-encoder calls).
- Running only BM25/vector produces noisy rankings that may not surface the best answer at position 1.
- Retrieval + reranking gives the best precision at manageable latency.

**The retrieve-then-rerank pattern** is the industry standard in production search systems (Google, Bing, Elasticsearch with learned ranking).

---

## 6. Failure modes

| Failure | Cause | Mitigation |
|---|---|---|
| Wrong document retrieved | Semantic drift in embeddings; BM25 keyword mismatch | Query transformation; corrective retrieval |
| Correct document retrieved, wrong answer extracted | Question phrasing differs from document language | Fallback to full context chunk; lower span threshold |
| Corrective retrieval makes things worse | Synonym expansion introduces noise | Only adopt if top_score strictly improves |
| High confidence, wrong answer | Cross-encoder promotes superficially similar but off-topic content | Human evaluation; calibration of score thresholds |
| Slow response | Large reranker candidate pool; slow embedding model | Reduce RERANK_TOP_K; use distilled models |
| No answer for valid question | Document not in the corpus | Extend the document collection; monitor "no answer" rate |

---

## 7. Scaling considerations

**20 documents (this project):**
- In-memory BM25, local ChromaDB, full cross-encoder reranking — no infrastructure needed.

**1 million documents:**
- BM25: persistent inverted index (Elasticsearch, OpenSearch, or Lucene).
- Vector: approximate nearest neighbour index (HNSW via FAISS or Qdrant) — exact NN search is O(n).
- Reranking: batch inference with GPU; limit to top-100 candidates from retrieval.
- ChromaDB is no longer sufficient — use a production vector store.

**100 million documents:**
- Sharded storage across multiple nodes.
- Distributed vector indexes with partition-level approximate search.
- Multi-stage retrieval: sparse → dense → reranker (each stage reduces candidate count).
- Asynchronous query pipelines with queuing.
- Embedding pre-computation with incremental updates.
- Cache hot queries; use approximate reranking for tail traffic.

---

## 8. Production monitoring metrics

A production RAG system needs continuous observability:

| Metric | Why it matters |
|---|---|
| Query latency (p50, p95, p99) | User experience; SLA compliance |
| Retrieval hit rate | % of queries where the correct document appears in top-k |
| Confidence level distribution | Monitor drift in HIGH/MEDIUM/LOW ratios — sudden drop signals corpus staleness |
| Corrective retrieval rate | High rate means the corpus or query patterns have changed |
| Answer fallback rate | % of queries returning "Insufficient evidence" — inverse proxy for system coverage |
| Reranker score distribution | Sudden shift in score ranges signals model drift or query distribution shift |
| Span extraction success rate | % of queries producing a specific answer span vs full-chunk fallback |
| End-to-end answer quality (sample) | Periodic human evaluation on random samples |

---

## 9. Retrieval failure examples

### A — BM25 succeeds, vector fails
**Query:** `"expense form EXP-2024-REV3"`  
BM25 matches the exact alphanumeric code (`EXP-2024-REV3`) via term lookup. Vector search compresses the code into an embedding that loses its distinguishing structure. **Lesson:** BM25 is irreplaceable for proper nouns, codes, and identifiers.

### B — Vector succeeds, BM25 fails
**Query:** `"time away from the office regulations"`  
BM25 tokenises to `["time", "away", "office", "regulations"]` — none of these match "leave policy" verbatim. The vector embedding of "time away from office" is semantically close to "leave policy" because the model has learned that both describe workplace absence. **Lesson:** Dense retrieval handles synonymy and paraphrase that BM25 cannot.

### C — Hybrid succeeds where either alone fails
**Query:** `"remote work approval and expense reimbursement"`  
Query decomposition produces two subqueries. BM25 retrieves `remote_work_policy` for subquery 1 and `expense_reimbursement` for subquery 2. Vector retrieval confirms both. RRF fusion surfaces both documents with high confidence. **Lesson:** Multi-topic queries require hybrid retrieval plus query decomposition to surface all relevant documents.

---

## 10. Common interview questions — foundational

1. **What is the primary benefit of hybrid retrieval?**
   Hybrid retrieval leverages lexical precision (BM25) and semantic generalisation (vector), improving recall across query types while maintaining precision via reranking.

2. **How does RRF work?**
   RRF computes `score += 1/(k + rank)` for each system. Documents ranked well by both systems receive additive boosts. It is scale-invariant — it uses only ranks, not raw scores from heterogeneous systems.

3. **Why use a cross-encoder instead of a bi-encoder for reranking?**
   Cross-encoders jointly encode the query-document pair, enabling full token-level attention interaction. Bi-encoders encode independently. Cross-encoders yield significantly higher ranking accuracy at the cost of O(n) inference calls.

4. **What is an extractive QA model's limitation?**
   It can only return spans that exist verbatim or near-verbatim in the context. It cannot generate summaries, reason across non-contiguous passages, or answer questions where the answer is implicit.

5. **Why is query transformation important?**
   Users write queries in different vocabularies than documents. Acronym expansion (MFA → multi-factor authentication), synonym normalisation, and typo correction bridge the vocabulary gap before retrieval begins.

6. **How should confidence be computed for cross-encoder scores?**
   Not by averaging — cross-encoder scores are ranking scores whose absolute values vary by model. The correct signals are: (1) top score (absolute relevance), (2) score margin (certainty gap to second result), (3) retrieval agreement (did multiple independent methods find this document?).

7. **What is a corrective retrieval loop?**
   A second retrieval pass triggered when initial confidence is low. It tries alternate query formulations (synonym expansion, simplification, broader depth) and adopts results only if they improve the top reranker score.

8. **Why not rerank all retrieved chunks?**
   Cross-encoders require one inference call per candidate pair. Reranking scales as O(n) in inference calls — running it over the full corpus would be prohibitively slow. Retrieval acts as a fast filter to reduce candidates to a manageable size.

9. **How do you ground answers with citations?**
   The extractive QA model returns the source document ID and chunk ID for the selected answer span. The UI displays this as a citation, allowing users to verify the answer against the original source.

10. **What would you monitor in production?**
    Query latency percentiles, confidence level distribution, corrective retrieval rate, answer fallback rate, reranker score distribution, end-to-end answer quality via periodic human evaluation.

---

## 11. Advanced interview questions — reranking

11. **Why does cross-encoder reranking improve over BM25/vector at the top of the ranking?**
    BM25 and vector retrieval use independent encodings that do not model direct query-document interaction. A cross-encoder reads the full concatenated [query; document] pair, letting every query token attend to every document token. This joint interaction captures relevance signals that independent encodings miss — for example, the query term appearing in a specific negated context in the document.

12. **What is the distillation trade-off when replacing a cross-encoder with a bi-encoder for speed?**
    A distilled bi-encoder (e.g., trained to mimic cross-encoder scores) loses ~5–15% in NDCG@10 on standard benchmarks. It reduces inference latency from O(n × model_size) to O(1) per query (the query is embedded once, then compared via fast ANN). The right choice depends on the latency budget: below 50ms SLAs typically require bi-encoders; above 200ms can afford cross-encoders.

13. **How would you evaluate whether your reranker is actually helping?**
    Compare NDCG@1, NDCG@5, MRR, and Precision@k before and after reranking on a held-out labelled query set. Also measure the rank correlation between pre-rerank and post-rerank orderings — low correlation with high NDCG improvement indicates the reranker is doing real work, not just confirming retrieval order.

14. **What is the failure mode of reranking with a small candidate pool?**
    If the correct document is not in the pre-rerank candidate pool, the reranker cannot surface it — it can only reorder what it receives. This is why retrieval recall (not precision) is the critical first-stage metric. You want the correct document in the top-20 retrieved, not necessarily at rank 1.

---

## 12. Advanced interview questions — hybrid search

15. **Why is BM25's k parameter important for hybrid systems?**
    The k₁ parameter in BM25 Okapi controls term frequency saturation (typically 1.2–2.0). Low k₁ means additional occurrences of a term provide diminishing returns quickly. For hybrid systems, k₁ affects the relative weight BM25 gives to term frequency vs document frequency — misconfigured k₁ can cause BM25 scores to be dominated by a single high-frequency term.

16. **When does RRF fail and what would you use instead?**
    RRF can fail when one retrieval system consistently produces irrelevant results at high rank — those results get undeserved boosts. Weighted RRF (assigning different weights to BM25 vs vector contributions) or learned rank fusion (using a small neural network trained on query-relevance labels) can address this. However, both require labelled data, which RRF does not.

17. **How do you handle embedding model staleness in production?**
    If new documents use vocabulary the embedding model hasn't seen, their embeddings may not align well with query embeddings. Mitigations: (1) periodic model fine-tuning on domain data, (2) monitoring OOV token rate in incoming queries, (3) BM25 as a safety net for exact keyword queries that don't depend on embeddings.

---

## 13. Advanced interview questions — confidence scoring

18. **Why is score margin more informative than raw score for retrieval confidence?**
    A large margin (e.g., top=3.98, second=-10.2, margin=14.18) indicates the retrieval is unambiguous — one document is clearly the best match. A small margin (e.g., top=1.5, second=1.3, margin=0.2) indicates the top two candidates are nearly equally relevant, meaning retrieval could easily have ranked them differently with a slightly different query or document representation.

19. **How do you calibrate confidence thresholds for a new document corpus?**
    Collect a set of labelled queries with known-correct documents. For each query, record the top reranker score and score margin. Plot the distribution of scores for correct vs incorrect top results. Set the HIGH threshold where precision is acceptably high (e.g., >90%), MEDIUM where precision is acceptable for fallback responses, and LOW where you'd rather not answer than risk incorrect information.

20. **What does retrieval agreement (BM25 + vector both finding the same document) signal?**
    Two independently operating retrieval systems using different information — lexical overlap vs semantic similarity — arriving at the same answer is strong evidence of relevance. It's analogous to independent corroboration in hypothesis testing. A document that satisfies both a keyword constraint and a semantic similarity constraint is more likely to be the correct answer than one satisfying only one.

---

## 14. Advanced interview questions — corrective retrieval

21. **What are the risks of query expansion in corrective retrieval?**
    (1) Semantic drift: adding synonyms can promote documents that are thematically related but not topically relevant. (2) Term pollution: adding many terms reduces BM25 IDF scores for all query terms. (3) Latency: each corrective attempt roughly doubles retrieval compute. Mitigations: expand conservatively, adopt only if improvement is strict, limit strategy count.

22. **How would you implement a query-difficulty predictor to avoid unnecessary corrective retrieval?**
    Train a small classifier on (query embedding, document collection statistics) → predicted confidence level. Features: query length, OOV rate, query-corpus vocabulary overlap, syntactic complexity. If the predictor flags a query as "likely low confidence", pre-emptively use broader retrieval depth on the first attempt rather than a second pass.

---

## 15. Advanced interview questions — evaluation metrics

23. **What is the difference between MRR, NDCG, and Recall@k, and when would you use each?**
    - **MRR** (Mean Reciprocal Rank): average of 1/rank of the first relevant document. Best when only the top-1 result matters (e.g., single-document QA).
    - **NDCG@k** (Normalised Discounted Cumulative Gain): rewards highly relevant documents at higher positions. Best for ranked lists where multiple documents can be relevant at different grades.
    - **Recall@k**: fraction of relevant documents appearing in the top-k. Best for measuring retrieval coverage — critical for the first-stage retrieval where recall matters more than precision.

24. **How would you build a QA evaluation dataset for this system without human annotation?**
    (1) Generate synthetic QA pairs from documents using an LLM (question from a passage, answer = the passage). (2) Use existing HR/policy QA benchmarks if available. (3) LLM-as-judge: use a larger model to rate answer quality for sampled queries. (4) Consistency checking: ask the same question multiple ways and flag when answers differ significantly.

25. **What is the RAGAS framework and how would you apply it here?**
    RAGAS (Retrieval Augmented Generation Assessment) measures: **faithfulness** (is the answer grounded in retrieved context?), **answer relevance** (does the answer address the question?), **context precision** (how much of the retrieved context is actually used?), and **context recall** (does retrieved context cover the answer?). Here, faithfulness is high by design (extractive QA) but answer relevance can fail when span extraction misidentifies the span.

---

## 16. How production RAG differs from this assignment

- Production systems separate ingestion, retrieval, and generation into independent services.
- Enterprise vector stores (Pinecone, Weaviate, Qdrant) provide horizontal scaling, RBAC, and versioned indexes.
- Production pipelines apply stronger access control, query logging, and auditability.
- Dynamic index updates (new documents added without full re-index) require incremental embedding + upsert pipelines.
- Response caching, query routing (route simple queries to fast paths, complex to deep pipelines), and circuit breakers are essential.
- Monitoring systems emit custom metrics to Datadog/Prometheus and trigger alerts on confidence drift, latency regression, and fallback rate increases.
- A/B testing infrastructure compares pipeline variants on live traffic before promotion.

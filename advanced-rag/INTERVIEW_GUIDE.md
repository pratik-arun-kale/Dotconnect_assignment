# Interview Guide for Advanced RAG System

## 1. Complete request flow

1. User enters a question in the Streamlit UI.
2. The query transformer decomposes compound questions into focused subqueries.
3. BM25 and vector retrieval are executed in parallel.
4. Results are fused using Reciprocal Rank Fusion (RRF).
5. The top candidates are reranked by a cross-encoder.
6. A retrieval confidence grade is computed.
7. If confidence is low, a corrective retrieval attempt is triggered.
8. Extractive QA extracts a grounded answer span from reranked chunks.
9. The UI displays the final answer, citations, and pipeline diagnostics.

## 2. Why not use only vector search?

- Vector search captures semantic similarity but can miss exact keyword matches.
- It may return false positives when embeddings overgeneralize.
- BM25 complements vector search with lexical precision.
- Hybrid systems are more robust across varied query types.

## 3. Why BM25?

- BM25 is an effective lexical retrieval baseline.
- It is fast, interpretable, and low-cost.
- It excels at keyword-driven queries and exact phrase matching.
- It provides a second signal that often improves precision.

## 4. Why Hybrid Search?

- Combining BM25 and vector search leverages the strengths of both.
- BM25 handles exact text matches; vector search handles semantics.
- Hybrid retrieval reduces the risk of missing relevant content.
- It improves recall without sacrificing relevancy.

## 5. Why RRF?

- Reciprocal Rank Fusion is a lightweight rank fusion method.
- It combines rankings rather than raw scores from heterogeneous systems.
- RRF is robust to score scale mismatches.
- It boosts documents that are strong in both retrieval modalities.

## 6. Why Cross Encoder Reranking?

- Cross-encoders jointly encode query and candidate text for deeper interaction.
- They outperform sparse and dense retrievers on final ranking quality.
- They are used only for a short candidate list to limit latency.
- They help suppress superficially similar but irrelevant results.

## 7. Why Retrieval Grading?

- Grading enables the system to detect low-confidence retrieval.
- It makes the pipeline adaptive instead of one-shot.
- It can trigger fallback paths when evidence is weak.
- It supports safer answer extraction with grounded sources.

## 8. Why Corrective Retrieval?

- The first retrieval attempt may miss the best evidence.
- Corrective retrieval broadens or expands the query when confidence is low.
- It prevents premature or hallucinated answers.
- It is a pragmatic tradeoff that improves robustness.

## 9. Why Extractive QA?

- Extractive QA returns a grounded text span from the source.
- It avoids generating unsupported claims.
- It is well-suited for factual QA over documents.
- It provides an evidence-based answer with a confidence score.

## 10. Why ChromaDB?

- ChromaDB is a lightweight local vector store.
- It supports persistence and embedding-based similarity search.
- It fits small-scale prototypes and proofs of concept.
- It keeps the project local without external APIs.

## 11. Latency analysis

- Embedding generation is performed during ingestion, not query time.
- BM25 retrieval is fast for small document collections.
- Vector retrieval is efficient on ChromaDB for 10-20 documents.
- Cross-encoder reranking is the largest latency contributor.
- Corrective retrieval doubles retrieval work in low-confidence cases.

## 12. Memory analysis

- Models like SentenceTransformer and CrossEncoder load into RAM.
- ChromaDB stores vectors on disk with in-memory acceleration.
- For 20 documents, memory use is modest.
- For larger collections, memory consumption grows with index size and model cache.

## 13. Production tradeoffs

- Local models are safer but more resource-intensive than API calls.
- Hybrid retrieval improves accuracy but adds complexity.
- Reranking boosts precision at the cost of extra compute.
- Extractive QA is safer than generation but may fail when no answer exists.
- Monitoring and logging become more important with more components.

## 14. Failure modes

- The query transformer may split a question incorrectly.
- BM25 may miss synonyms or paraphrases.
- Vector retrieval may return semantically related but irrelevant passages.
- Reranker may still promote a wrong candidate if the top candidates are poor.
- Extractive QA may return an empty span or low-confidence answer.

## 15. Monitoring metrics

- Query latency: total and per-stage.
- Candidate retrieval counts and scores.
- Reranker average score and distribution.
- QA confidence and answer fallback rates.
- Corrective retrieval trigger rate.

## 16. Scaling from 20 docs -> 1M docs -> 100M docs

- 20 docs: prototype, in-memory BM25, local ChromaDB, full cross-encoder reranking.
- 1M docs: need persistent inverted index, approximate nearest neighbor search, batch retrieval, and staged reranking.
- 100M docs: require sharded storage, distributed vector indexes, asynchronous query pipelines, and multi-stage filtering.

## 17. How production RAG differs from this assignment

- Production systems often separate retrieval and generation services.
- They use enterprise vector stores and indexing infrastructure.
- They apply stronger access control, monitoring, and auditability.
- They support dynamic index updates, caching, and load balancing.
- They often use prompt templates or retrieval-augmented generative models rather than pure extractive QA.

## 18. Common interview questions and answers

1. **What is the primary benefit of hybrid retrieval?**
   Hybrid retrieval leverages both lexical and semantic signals, improving recall and reducing reliance on a single retrieval modality.

2. **How does RRF work?**
   RRF uses reciprocal rank scores from multiple retrieval lists to combine results, giving higher fused score to candidates ranked well by multiple systems.

3. **Why use a cross-encoder instead of a bi-encoder for reranking?**
   Cross-encoders model interactions between the query and text directly, yielding higher accuracy on ranking smaller candidate sets.

4. **What is an extractive QA model's limitation?**
   It can only answer questions whose answer exists verbatim or near-verbatim in the context; it cannot generate novel summaries.

5. **Why is query transformation important?**
   Transforming a compound query into subqueries improves retrieval focus and reduces confusion in ranking and answer extraction.

6. **How is retrieval confidence computed?**
   Confidence is computed as an average of reranker relevance scores, which indicates the quality of the top-ranked candidates.

7. **What is a corrective retrieval loop?**
   It is a second retrieval pass triggered when initial results are low confidence, often using query expansion or broader search.

8. **Why not rerank all retrieved chunks?**
   Reranking is expensive, so it is applied only to a short candidate list after retrieval fusion.

9. **How do you ground answers with citations?**
   By returning the source document and chunk id for the selected answer span, allowing users to verify evidence.

10. **What is the role of BM25 here?**
    BM25 supplies a lexical signal that is especially strong for keyword-specific questions.

11. **What would you monitor in a production retrieval system?**
    Latency, hit rate, reranker performance, query distribution, and fallback frequency.

12. **How can you extend this system to support PDFs?**
    Add a loader for PDF files, convert pages to text, and chunk them similarly before indexing.

13. **Why is local execution valuable in interviews?**
    It demonstrates end-to-end ownership, reproducibility, and the ability to manage models and indexing without black-box APIs.

14. **How would you improve synonym coverage?**
    Use a small lexical knowledge base, word embeddings, or offline thesaurus expansion rather than relying on a single dictionary.

15. **What are the tradeoffs of using a larger reranker?**
    Better ranking quality at higher latency and memory cost.

16. **What is the difference between retrieval and generation?**
    Retrieval finds relevant evidence; generation produces natural language, often conditioned on the retrieved evidence.

17. **Why not rely on a generative model for answers?**
    Generative models can hallucinate; extractive QA maintains stronger evidence alignment.

18. **How can you measure if reranking helps?**
    Compare precision@k and recall@k before and after reranking on labeled queries.

19. **What could cause low retrieval confidence?**
    Ambiguous queries, poor candidate quality, or missing coverage in the document collection.

20. **Why is query expansion useful?**
    It helps cover synonyms and alternate phrasing that initial retrieval missed.

21. **How do you handle negation in retrieval?**
    Query parsing should preserve negation terms and avoid oversimplifying the question when splitting or expanding.

22. **What is the downside of simple rule-based query transformation?**
    It can over-split or miss complex semantic structure.

23. **How should you handle an unanswered question?**
    Return an explicit fallback such as "Insufficient evidence found in retrieved documents." and avoid fabricating responses.

24. **What metrics matter for QA evaluation?**
    Exact match, F1 score, answer recall, and human verification of citation correctness.

25. **Why use ChromaDB instead of raw embeddings?**
    ChromaDB provides persistence, efficient similarity search, and integration with vector retrieval APIs.

26. **How would you scale QA for 100 million documents?**
    Use document filtering, sparse retrieval, approximate nearest neighbors, and a multi-stage reranking pipeline.

27. **What is the effect of chunk size on retrieval?**
    Small chunks increase precision but may lose context; large chunks increase recall but may dilute relevance.

28. **How do you preserve chunk metadata?**
    Store document IDs and chunk IDs in the vector store metadata and include them in results.

29. **What is the role of the embedding model?**
    It maps text to dense vectors that capture semantic similarity for vector retrieval.

30. **How can this system be productionized?**
    Add automated ingestion, versioned indexes, API endpoints, monitoring, caching, and a distributed retrieval architecture.

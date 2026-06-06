# Architecture

## 1. End-to-End Flow

1. Document ingestion:
   - Load raw text files from `data/`.
   - Split documents into overlapping chunks.
   - Generate vector embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
   - Persist chunks and vectors into ChromaDB.

2. Query processing:
   - Accept a user query in Streamlit.
   - Apply rule-based query transformation to split complex questions into focused subqueries.

3. Retrieval pipeline:
   - Run BM25 over chunk text.
   - Run vector similarity search over the ChromaDB index.
   - Fuse BM25 and vector results using Reciprocal Rank Fusion (RRF).
   - Rerank fused candidates with a cross-encoder.

4. Answer extraction:
   - Use extractive QA (`deepset/roberta-base-squad2`) to find grounded spans in top reranked chunks.
   - Return a concise answer with citation metadata.

5. Corrective retrieval loop:
   - Grade the top reranked results using average reranker score.
   - If confidence is low, broaden the query with synonym expansion and run a second retrieval attempt.

## 2. Retrieval Pipeline

- **BM25 Retrieval**
  - Lightweight lexical matching.
  - Good at exact keyword hits and high precision for specific terms.

- **Vector Retrieval**
  - Semantic matching using dense embeddings.
  - Captures meaning even when query terms differ from document text.

- **RRF Fusion**
  - Combines BM25 and vector ranks into a single score.
  - Mitigates weaknesses from either retrieval signal individually.

## 3. Hybrid Retrieval Design

Hybrid retrieval is implemented in `retrieval/rrf.py` and used by the UI pipeline.

- Retrieve top candidates separately from BM25 and vector search.
- Represent each candidate with both retrieval ranks.
- Compute a fused score using:
  - `1 / (k + rank)` for each retrieval list.
- Return the best combined candidates for reranking.

## 4. Reranking Design

- Use `cross-encoder/ms-marco-MiniLM-L-6-v2` to score query-chunk pairs.
- Reranker sorts the top fused candidates and selects the most relevant slices.
- Cross-encoder improves precision by considering interaction between query and chunk.

## 5. Corrective Loop Design

- Grade retrieval quality using average reranker score.
- If the average falls below a threshold, apply simple query expansion.
- Run retrieval again with the broader query.
- Limit to at most two retrieval attempts.

## 6. Citation System

- Each answer includes:
  - `source_doc`
  - `chunk_id`
- Citations are built in `generation/citation_builder.py`.
- Grounded answers allow users to verify source context.

## 7. Sequence Diagrams

### User Query

User -> Streamlit UI -> Query Transformer -> Retrieval Engine -> Reranker -> QA Extractor -> Streamlit UI

### Retrieval Loop

Query -> BM25 + Vector -> RRF -> Rerank -> Grade -> [if low] Expand query -> BM25 + Vector -> RRF -> Rerank -> Answer

## 8. Scaling Discussion

- At 20 documents, all components run locally with low latency.
- At 1M documents, vector search and BM25 require production index structures, more memory, and distributed storage.
- At 100M documents, use sharding, approximate nearest neighbors, and enterprise retrieval services.

### Scaling considerations

- **ChromaDB**: good prototype storage. Production may require a vector database optimized for scale.
- **BM25**: rebuild and persist inverted indexes rather than tokenizing at runtime.
- **Reranking**: apply only to top-K candidates after retrieval to control latency.
- **QA**: batch candidate contexts for efficient processing.
- **Monitoring**: track query latency, precision, recall, and model drift.

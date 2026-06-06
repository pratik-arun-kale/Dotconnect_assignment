# Advanced RAG System

This repository contains a locally executable, interview-quality Retrieval-Augmented Generation (RAG) system built in Python. It demonstrates advanced retrieval features beyond naive vector search using a small document collection, with a Streamlit frontend for exploration.

## Problem Statement

Build a question answering system over a small document collection that goes beyond simple vector search by incorporating query transformation, hybrid retrieval, reranking, grounded extraction, citations, and a corrective retrieval loop.

## Architecture

- `ingestion/`: load documents, chunk content, generate embeddings, and build a ChromaDB vector index.
- `retrieval/`: perform BM25 retrieval, vector retrieval, reciprocal rank fusion, reranking, and retrieval grading.
- `generation/`: extract answers with an extractive QA model and format citations.
- `ui/`: Streamlit application that exposes the pipeline with professional debugging insights.
- `data/`: sample document collection for demonstration.

## Installation

1. Create a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
cd advanced-rag
python -m pip install -r requirements.txt
```

> Note: Streamlit can be sensitive to the installed Starlette version. The pinned requirements include `streamlit==1.24.1` and `starlette<0.28` to avoid the `DEFAULT_EXCLUDED_CONTENT_TYPES` import error.

## Execution

1. Build the index:

```bash
python ingestion/build_index.py
```

2. Launch the Streamlit UI:

```bash
streamlit run ui/streamlit_app.py
```

## Example Queries

- `What are the remote work requirements and who approves remote work?`
- `How do employees request expense reimbursement?`
- `What are the security obligations for company laptops?`

## Screenshots

- Placeholder: Streamlit dashboard with retrieval stages, reranker scores, confidence, and citations.

## Tradeoffs

- The system uses a hybrid retrieval pipeline to reduce over-reliance on a single signal.
- Cross-encoder reranking adds latency but improves precision on the final answer set.
- Extractive QA is grounded in retrieved context, avoiding hallucinations.
- Corrective retrieval trades additional compute for robustness when initial confidence is low.

## Future Improvements

- Add support for PDF and DOCX ingest.
- Persist BM25 state so retrieval does not rebuild on every run.
- Add caching for reranker results and query expansions.
- Introduce an index refresh workflow and monitoring service.

## Evaluation Results

The evaluation module includes a basic retrieval test harness. For production, add labeled query-document pairs and metrics such as recall@k, precision@k, and exact match.

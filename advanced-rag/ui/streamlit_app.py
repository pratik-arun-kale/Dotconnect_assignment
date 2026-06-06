import logging
import sys
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
from retrieval.query_transform import transform_query
from retrieval.reranker import Reranker
from retrieval.retrieval_grader import expand_query, grade_retrieval
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.vector_retriever import VectorRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    bm25_retriever = BM25Retriever([{"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata} for chunk in chunks])
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


def run_single_query(query: str, pipeline: Dict) -> Dict:
    bm25 = pipeline["bm25"]
    vector = pipeline["vector"]
    reranker = pipeline["reranker"]
    qa = pipeline["qa"]

    transformed_queries = transform_query(query)
    responses = []
    for subquery in transformed_queries:
        bm25_results = bm25.retrieve(subquery, top_k=VECTOR_TOP_K)
        vector_results = vector.retrieve(subquery, top_k=VECTOR_TOP_K)
        fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=FUSED_TOP_K)
        reranked = reranker.rerank(subquery, fused, top_k=RERANK_TOP_K)
        confidence, needs_correction = grade_retrieval(reranked)
        attempt = 1
        query_used = subquery
        corrective_status = "Not triggered"

        if needs_correction and attempt < MAX_RETRIEVAL_ATTEMPTS:
            attempt += 1
            expanded = expand_query(subquery)
            bm25_results = bm25.retrieve(expanded, top_k=VECTOR_TOP_K)
            vector_results = vector.retrieve(expanded, top_k=VECTOR_TOP_K)
            fused = reciprocal_rank_fusion(bm25_results, vector_results, top_k=FUSED_TOP_K)
            reranked = reranker.rerank(expanded, fused, top_k=RERANK_TOP_K)
            expanded_confidence, _ = grade_retrieval(reranked)
            if expanded_confidence >= confidence:
                confidence = expanded_confidence
                query_used = expanded
            corrective_status = "Corrective retrieval triggered"

        answer = qa.answer_question(subquery, reranked)
        citations = build_citations([answer])

        responses.append(
            {
                "subquery": subquery,
                "query_used": query_used,
                "bm25_results": bm25_results,
                "vector_results": vector_results,
                "fused_results": fused,
                "reranked_results": reranked,
                "confidence": confidence,
                "corrective_status": corrective_status,
                "attempt_count": attempt,
                "answer": answer,
                "citations": citations,
            }
        )

    return {
        "original_query": query,
        "transformed_queries": transformed_queries,
        "responses": responses,
    }


def render_results(output: Dict) -> None:
    st.subheader("Query Results")
    st.write(f"**Original query:** {output['original_query']}")
    st.write("**Transformed subqueries:**")
    for subquery in output["transformed_queries"]:
        st.write(f"- {subquery}")

    for response in output["responses"]:
        with st.expander(f"Subquery: {response['subquery']}"):
            st.markdown(f"**Query used for retrieval:** {response['query_used']}")
            st.markdown(f"**Retrieval confidence:** {response['confidence']:.3f}")
            st.markdown(f"**Corrective loop:** {response['corrective_status']} (attempts: {response['attempt_count']})")

            st.markdown("**Final Extractive Answer**")
            st.write(response["answer"]["answer"])
            if response["citations"]:
                st.markdown("**Citations**")
                for citation in response["citations"]:
                    st.write(f"- {citation}")

            st.markdown("**BM25 Top Results**")
            for item in response["bm25_results"]:
                st.write(f"{item['rank']}. {item['metadata']['doc_id']} | {item['chunk_id']} | score: {item['score']:.3f}")

            st.markdown("**Vector Top Results**")
            for item in response["vector_results"]:
                st.write(f"{item['rank']}. {item['metadata']['doc_id']} | {item['chunk_id']} | score: {item['score']:.3f}")

            st.markdown("**RRF Fused Results**")
            for item in response["fused_results"]:
                st.write(
                    f"{item['fused_rank']}. {item['metadata']['doc_id']} | {item['chunk_id']} | bm25_rank: {item['bm25_rank']} | vector_rank: {item['vector_rank']} | rrf: {item['rrf_score']:.4f}"
                )

            st.markdown("**Reranked Results**")
            for item in response["reranked_results"]:
                st.write(
                    f"{item['reranked_rank']}. {item['metadata']['doc_id']} | {item['chunk_id']} | reranker_score: {item['reranker_score']:.4f}"
                )


def main() -> None:
    st.set_page_config(page_title="Advanced RAG System", layout="wide")
    st.title("Advanced Retrieval-Augmented Generation (RAG) Demo")
    st.markdown(
        "Use this interface to explore how query transformation, hybrid retrieval, reranking, grounded QA, and corrective retrieval work together."
    )

    query = st.text_input("Enter a question for the document collection:")
    if not query:
        st.info("Type a question above to begin retrieval and answer extraction.")
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

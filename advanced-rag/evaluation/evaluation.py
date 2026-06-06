import logging
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ingestion.build_index import main as build_index
from ingestion.chunker import Chunker
from ingestion.loader import DocumentLoader
from retrieval.bm25_retriever import BM25Retriever
from retrieval.query_transform import transform_query
from retrieval.reranker import Reranker
from retrieval.retrieval_grader import grade_retrieval, expand_query
from retrieval.rrf import reciprocal_rank_fusion
from retrieval.vector_retriever import VectorRetriever
from generation.extractive_qa import ExtractiveQA
from ingestion.index_builder import IndexBuilder

logger = logging.getLogger(__name__)

SAMPLE_QUERIES = [
    "What is the remote work policy for approval?",
    "How do employees request expense reimbursement?",
    "What are the acceptable use requirements for company devices?",
]


def run_evaluation() -> None:
    loader = DocumentLoader()
    documents = loader.load_documents()
    if not documents:
        logger.error("No documents available for evaluation.")
        return

    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)
    index_builder = IndexBuilder()
    collection = index_builder.load_collection()

    bm25 = BM25Retriever([{"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata} for chunk in chunks])
    vector = VectorRetriever(collection)
    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    qa = ExtractiveQA("deepset/roberta-base-squad2")

    results: List[Dict] = []
    for query in SAMPLE_QUERIES:
        subqueries = transform_query(query)
        for subquery in subqueries:
            bm25_results = bm25.retrieve(subquery)
            vector_results = vector.retrieve(subquery)
            fused = reciprocal_rank_fusion(bm25_results, vector_results)
            reranked = reranker.rerank(subquery, fused)
            top_score, score_margin, confidence_level, needs_correction, reason = grade_retrieval(reranked)
            answer = qa.answer_question(subquery, reranked, fallback_to_context=(confidence_level != "LOW")) if confidence_level != "LOW" else {
                "answer": "Insufficient evidence found in retrieved documents.",
                "score": 0.0, "source_doc": None, "chunk_id": None,
            }
            results.append({
                "query": query,
                "subquery": subquery,
                "top_score": top_score,
                "score_margin": score_margin,
                "confidence_level": confidence_level,
                "confidence_reason": reason,
                "needs_correction": needs_correction,
                "answer": answer["answer"],
                "citation": f"{answer.get('source_doc')} | {answer.get('chunk_id')}" if answer.get("source_doc") else "None",
            })

    for row in results:
        print(row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()

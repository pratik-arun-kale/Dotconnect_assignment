from typing import Dict, List


def reciprocal_rank_fusion(
    bm25_results: List[Dict],
    vector_results: List[Dict],
    top_k: int = 20,
    k: int = 60,
) -> List[Dict]:
    scores = {}
    for result in bm25_results:
        chunk_id = result["chunk_id"]
        scores.setdefault(chunk_id, {
            "chunk_id": chunk_id,
            "text": result["text"],
            "metadata": result["metadata"],
            "bm25_rank": result["rank"],
            "vector_rank": None,
            "rrf_score": 0.0,
        })
        scores[chunk_id]["rrf_score"] += 1.0 / (k + result["rank"])

    for result in vector_results:
        chunk_id = result["chunk_id"]
        entry = scores.setdefault(chunk_id, {
            "chunk_id": chunk_id,
            "text": result["text"],
            "metadata": result["metadata"],
            "bm25_rank": None,
            "vector_rank": result["rank"],
            "rrf_score": 0.0,
        })
        entry["vector_rank"] = result["rank"]
        entry["rrf_score"] += 1.0 / (k + result["rank"])

    fused = sorted(scores.values(), key=lambda item: item["rrf_score"], reverse=True)
    for index, item in enumerate(fused[:top_k], start=1):
        item["fused_rank"] = index
    return fused[:top_k]

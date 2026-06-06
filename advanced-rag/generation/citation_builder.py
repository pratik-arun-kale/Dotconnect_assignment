from typing import List, Dict


def build_citations(answer_records: List[Dict]) -> List[str]:
    citations = []
    for record in answer_records:
        if record.get("source_doc") and record.get("chunk_id"):
            citations.append(f"{record['source_doc']} | {record['chunk_id']}")
    return citations

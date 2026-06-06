import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)


class ExtractiveQA:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        logger.info("Loaded extractive QA model %s", model_name)

    def _predict(self, question: str, context: str) -> Dict:
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            stride=128,
            return_overflowing_tokens=False,
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_idx = int(torch.argmax(start_logits))
        end_idx = int(torch.argmax(end_logits))

        if end_idx < start_idx:
            end_idx = start_idx

        input_ids = inputs["input_ids"][0]
        answer_tokens = input_ids[start_idx : end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        start_score = float(torch.max(torch.softmax(start_logits, dim=0)))
        end_score = float(torch.max(torch.softmax(end_logits, dim=0)))
        score = (start_score + end_score) / 2.0

        return {"answer": answer, "score": score}

    def answer_question(self, question: str, candidates: List[Dict]) -> Dict:
        best_response: Optional[Dict] = None
        for candidate in candidates:
            try:
                response = self._predict(question, candidate["text"])
            except Exception as exc:
                logger.warning("QA model failed on candidate %s: %s", candidate.get("chunk_id"), exc)
                continue

            if not response or response.get("score", 0.0) <= 0.0:
                continue

            response_data = {
                "answer": response["answer"],
                "score": response["score"],
                "source_doc": candidate["metadata"].get("doc_id"),
                "chunk_id": candidate["chunk_id"],
                "context": candidate["text"],
            }
            if not best_response or response_data["score"] > best_response["score"]:
                best_response = response_data

        if not best_response or best_response["score"] < 0.15 or not best_response["answer"]:
            logger.info("No confident extractive answer found for question: %s", question)
            return {
                "answer": "Insufficient evidence found in retrieved documents.",
                "score": 0.0,
                "source_doc": None,
                "chunk_id": None,
                "context": None,
            }

        logger.debug("Selected answer with score %.4f", best_response["score"])
        return best_response

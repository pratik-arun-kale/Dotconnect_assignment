import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

logger = logging.getLogger(__name__)

# Minimum extractive-model confidence to trust a found span.
# Only applied when the caller does NOT request a context fallback —
# see answer_question(fallback_to_context=True).
_SPAN_SCORE_THRESHOLD = 0.15


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

        # Score: use sigmoid on the best-span combined logit rather than softmax
        # over all positions.  Softmax distributes probability mass across every
        # token in a 512-token sequence, making the per-position score very small
        # even when the model is confident.  Sigmoid on the sum of the best
        # start/end logits gives a score that reflects model confidence without
        # that dilution effect.
        best_span_logit = (start_logits[start_idx] + end_logits[end_idx]).item()
        score = float(torch.sigmoid(torch.tensor(best_span_logit / 2.0)))

        return {"answer": answer, "score": score}

    def answer_question(
        self,
        question: str,
        candidates: List[Dict],
        fallback_to_context: bool = False,
    ) -> Dict:
        """
        Extract the best answer span from candidates.

        fallback_to_context=True: when no confident span is found and the caller
        already has HIGH/MEDIUM retrieval confidence, return the top candidate's
        full text rather than "Insufficient evidence".  The caller is responsible
        for deciding when this is appropriate.
        """
        if candidates:
            logger.info(
                "TOP DOCUMENT (chunk_id=%s):\n%s",
                candidates[0].get("chunk_id", "?"),
                candidates[0]["text"],
            )

        best_response: Optional[Dict] = None
        for candidate in candidates:
            try:
                response = self._predict(question, candidate["text"])
            except Exception as exc:
                logger.warning(
                    "QA model failed on candidate %s: %s",
                    candidate.get("chunk_id"),
                    exc,
                )
                continue

            if not response or response.get("score", 0.0) <= 0.0:
                continue

            logger.info(
                "Candidate %s → answer=%r score=%.4f",
                candidate.get("chunk_id"),
                response.get("answer", ""),
                response.get("score", 0.0),
            )

            response_data = {
                "answer": response["answer"],
                "score": float(response["score"]),
                "source_doc": candidate["metadata"].get("doc_id"),
                "chunk_id": candidate["chunk_id"],
                "context": candidate["text"],
            }
            if not best_response or response_data["score"] > best_response["score"]:
                best_response = response_data

        span_found = (
            best_response is not None
            and best_response["score"] >= _SPAN_SCORE_THRESHOLD
            and best_response["answer"]
        )

        if span_found:
            logger.info(
                "Extractive answer selected: %r (score=%.4f)",
                best_response["answer"],
                best_response["score"],
            )
            return best_response

        # No confident span found.
        if fallback_to_context and candidates:
            # The reranker already confirmed the top document is highly relevant,
            # so return its full text as a passage answer rather than silence.
            top = candidates[0]
            logger.info(
                "Span extraction failed; falling back to top chunk context (chunk_id=%s)",
                top.get("chunk_id"),
            )
            return {
                "answer": top["text"],
                "score": 0.0,
                "source_doc": top["metadata"].get("doc_id"),
                "chunk_id": top["chunk_id"],
                "context": top["text"],
            }

        logger.info("No confident extractive answer found for question: %s", question)
        return {
            "answer": "Insufficient evidence found in retrieved documents.",
            "score": 0.0,
            "source_doc": None,
            "chunk_id": None,
            "context": None,
        }

from __future__ import annotations

import logging

import torch

from .prompts import build_judge_prompt

log = logging.getLogger(__name__)

_IDK_PHRASES = [
    "i don't know",
    "i do not know",
    "the context does not",
    "does not provide enough information",
    "cannot answer",
    "cannot provide",
    "no information",
]


def _substring_is_refusal(response: str) -> bool:
    lower = response.lower()
    return any(phrase in lower for phrase in _IDK_PHRASES)


class LocalJudge:
    """
    Binary judge: did the RAG system answer the query?

    Replaces Shafran's GPT-4 judge with a local Gemma-2-9B-it.
    Fast path: substring check for common refusal phrases.
    Slow path: Gemma-2-9B with few-shot prompt → parse YES/NO.

    Load via HuggingFace (not vLLM) to avoid conflicts with the target LLM.
    Must be loaded AFTER the target LLM is closed (GPUManager constraint).
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-9b-it",
        device: str | None = None,
        max_new_tokens: int = 4,
    ) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        log.info("Loading judge: %s", model_name)
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._model.eval()
        log.info("Judge loaded.")

    def is_answered(self, query: str, response: str) -> bool:
        """
        Returns True if the RAG system answered the query.
        Returns False if it refused or said "I don't know".
        """
        # Fast path: substring check (no GPU needed)
        if _substring_is_refusal(response):
            return False

        # Slow path: LLM judge
        prompt = build_judge_prompt(query, response)
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tok.eos_token_id,
            )
        generated = self._tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().upper()
        return generated.startswith("YES")

    def is_answered_batch(
        self, queries: list[str], responses: list[str]
    ) -> list[bool]:
        return [self.is_answered(q, r) for q, r in zip(queries, responses)]

    def close(self) -> None:
        import gc
        log.info("Unloading judge.")
        del self._model
        del self._tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class VLLMGenerator:
    """
    vLLM wrapper for the target LLM.

    Registers itself with GPUManager under the key 'target_llm'.
    vLLM manages its own CUDA memory; call close() before loading other
    large models (judge, oracle embedder).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        gpu_memory_utilization: float = 0.85,
        dtype: str = "float16",
        max_model_len: int = 4096,
    ) -> None:
        from vllm import LLM, SamplingParams  # deferred so tests can mock

        self.model_name = model_name
        self._sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        log.info("Loading vLLM model: %s (utilization=%.0f%%)", model_name,
                 gpu_memory_utilization * 100)
        self._llm = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
        )
        log.info("vLLM model loaded.")

    def generate(self, prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        outputs = self._llm.generate(prompts, self._sampling_params)
        return [o.outputs[0].text for o in outputs]

    def close(self) -> None:
        """
        Destroy vLLM engine and release GPU memory.
        Must be called before loading another large model.
        """
        import gc
        import torch

        log.info("Closing vLLM engine for %s", self.model_name)
        del self._llm
        self._llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

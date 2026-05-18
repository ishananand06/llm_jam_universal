from __future__ import annotations

import gc
import logging
from typing import Callable, TypeVar

import torch

log = logging.getLogger(__name__)

M = TypeVar("M")


class GPUManager:
    """
    Tracks models loaded onto the GPU and provides explicit load/unload.

    Single-GPU constraint: the L40S has 46 GB shared across this process.
    Keeping multiple large models resident simultaneously causes OOM.
    Always unload a model before loading the next large one.

    Usage:
        gpu = GPUManager()
        model = gpu.load("retriever", lambda: SentenceTransformer(...))
        # ... use model ...
        gpu.unload("retriever")

    Or as a context manager (auto-unloads everything on exit):
        with GPUManager() as gpu:
            model = gpu.load("retriever", lambda: SentenceTransformer(...))
    """

    def __init__(self) -> None:
        self._registry: dict[str, object] = {}

    def load(self, name: str, loader_fn: Callable[[], M]) -> M:
        if name in self._registry:
            log.debug("GPUManager: '%s' already loaded, returning cached.", name)
            return self._registry[name]  # type: ignore[return-value]
        log.info("GPUManager: loading '%s'", name)
        model = loader_fn()
        self._registry[name] = model
        self._log_vram()
        return model

    def unload(self, name: str) -> None:
        if name not in self._registry:
            log.debug("GPUManager: '%s' not loaded, nothing to unload.", name)
            return
        log.info("GPUManager: unloading '%s'", name)
        obj = self._registry.pop(name)
        del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log_vram()

    def unload_all(self) -> None:
        for name in list(self._registry.keys()):
            self.unload(name)

    def loaded_models(self) -> list[str]:
        return list(self._registry.keys())

    def get(self, name: str) -> object | None:
        return self._registry.get(name)

    def _log_vram(self) -> None:
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            log.info("GPUManager: VRAM allocated=%.1f GB, reserved=%.1f GB", used, reserved)

    def __enter__(self) -> "GPUManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.unload_all()

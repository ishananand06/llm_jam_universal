from __future__ import annotations

import logging
from dataclasses import dataclass, field

from omegaconf import DictConfig

from .retriever import Retriever
from .generator import VLLMGenerator

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    query: str
    response: str
    retrieved_doc_ids: list[str] = field(default_factory=list)
    context: str = ""


class RAGPipeline:
    """
    End-to-end RAG: retrieve top-k docs, format context, generate response.

    The adversarial doc (if provided) is inserted at the start of the
    retrieved context so it appears first — simulating injection into the
    knowledge base.
    """

    def __init__(
        self,
        retriever: Retriever,
        generator: VLLMGenerator,
        cfg: DictConfig,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.cfg = cfg

    def _format_context(self, doc_texts: list[str]) -> str:
        return "\n\n".join(doc_texts)

    def _format_prompt(self, context: str, query: str) -> str:
        return self.cfg.rag_prompt.format(context=context, query=query)

    def run(
        self,
        query: str,
        adv_doc: str | None = None,
        k: int | None = None,
    ) -> PipelineResult:
        k = k or self.cfg.retrieval.k
        doc_ids = self.retriever.retrieve(query, k=k)
        doc_texts = [self.retriever.get_doc_text(d) for d in doc_ids]

        if adv_doc is not None:
            doc_texts = [adv_doc] + doc_texts[:k - 1]

        context = self._format_context(doc_texts)
        prompt = self._format_prompt(context, query)
        responses = self.generator.generate([prompt])
        return PipelineResult(
            query=query,
            response=responses[0],
            retrieved_doc_ids=doc_ids,
            context=context,
        )

    def run_batch(
        self,
        queries: list[str],
        adv_docs: list[str | None] | None = None,
        k: int | None = None,
    ) -> list[PipelineResult]:
        k = k or self.cfg.retrieval.k
        if adv_docs is None:
            adv_docs = [None] * len(queries)

        prompts: list[str] = []
        meta: list[tuple[str, list[str], str]] = []  # (query, doc_ids, context)

        for query, adv_doc in zip(queries, adv_docs):
            doc_ids = self.retriever.retrieve(query, k=k)
            doc_texts = [self.retriever.get_doc_text(d) for d in doc_ids]
            if adv_doc is not None:
                doc_texts = [adv_doc] + doc_texts[:k - 1]
            context = self._format_context(doc_texts)
            prompts.append(self._format_prompt(context, query))
            meta.append((query, doc_ids, context))

        responses = self.generator.generate(prompts)
        return [
            PipelineResult(query=m[0], response=r, retrieved_doc_ids=m[1], context=m[2])
            for m, r in zip(meta, responses)
        ]

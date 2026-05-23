from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # models
    dual_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # stage parameters
    top_n: int = 20       # dual-encoder candidates before expansion
    hop_depth: int = 1    # AST neighbour hops (0 = no expansion)
    top_k: int = 5        # final results after cross-encoder rerank
    # query prefix — prepended to the query string before encoding.
    # Leave "" for MiniLM-style models.
    # Use "query: " for BAAI/bge-m3 and most BGE-family models.
    # Use "Represent this sentence for searching relevant passages: "
    # for older bge-large-en-v1.5.
    query_prefix: str = ""
    # runtime
    batch_size: int = 32
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @classmethod
    def for_bge_m3(cls, **kwargs) -> "PipelineConfig":
        """Convenience constructor with BGE-M3 defaults."""
        return cls(
            dual_encoder_model="BAAI/bge-m3",
            query_prefix="query: ",
            batch_size=16,   # BGE-M3 is larger, smaller batch to fit GPU
            **kwargs,
        )

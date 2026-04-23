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

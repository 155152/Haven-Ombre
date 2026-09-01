#!/usr/bin/env python3
from __future__ import annotations

import os

import torch
from sentence_transformers import CrossEncoder


MODEL = os.environ.get("OMBRE_LOCAL_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
DEVICE = os.environ.get("OMBRE_LOCAL_RERANKER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = int(os.environ.get("OMBRE_LOCAL_RERANKER_MAX_LENGTH", "2048"))


def main() -> int:
    print(f"preparing reranker model={MODEL} device={DEVICE}")
    model = CrossEncoder(MODEL, device=DEVICE, max_length=MAX_LENGTH)
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Mars is often called the Red Planet because of its reddish appearance.",
        "The Pacific Ocean is the largest ocean on Earth.",
    ]
    scores = model.predict(
        [(query, document) for document in documents],
        activation_fn=torch.nn.Sigmoid(),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    relevant = float(scores[0])
    irrelevant = float(scores[1])
    print(f"relevant={relevant:.6f} irrelevant={irrelevant:.6f}")
    if relevant <= irrelevant:
        raise RuntimeError("reranker smoke test failed: relevant passage did not outrank irrelevant passage")
    print("reranker ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

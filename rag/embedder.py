"""all-MiniLM-L6-v2 embeddings on the GPU, or on the CPU when the GPU fails."""
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self._load("cuda" if torch.cuda.is_available() else "cpu", fall_back=True)

    _cpu_model = None

    def _load(self, device, fall_back=False):
        try:
            self.model = SentenceTransformer(MODEL_NAME, device=device)
        except RuntimeError as error:
            if not fall_back or device == "cpu":
                raise
            self._use_cpu(error)
            return
        self.device = device
        # The model card default of 256 would cut 500-token chunks in half.
        self.model.max_seq_length = 512
        self.tokenizer = self.model.tokenizer

    def _use_cpu(self, error):
        log.warning("GPU failed, using the CPU for embeddings: %s", str(error).splitlines()[0])
        # The failed model may be stuck on a broken GPU context, so load a new one.
        self._load("cpu")

    @staticmethod
    def _run(model, texts):
        return np.asarray(model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        ), dtype=float)

    def _cpu(self):
        """A CPU copy for the odd vector the GPU gets wrong; the GPU keeps doing the rest."""
        if self._cpu_model is None:
            self._cpu_model = SentenceTransformer(MODEL_NAME, device="cpu")
            self._cpu_model.max_seq_length = 512
        return self._cpu_model

    def _encode(self, texts):
        vectors = self._run(self.model, texts)
        bad = [i for i, row in enumerate(vectors) if not np.isfinite(row).all()]
        if bad and self.device != "cpu":
            # The GPU now and then returns NaN for a text (11/2026), which pgvector refuses.
            log.warning("GPU gave %d vector(s) that are not numbers, computing them on the CPU", len(bad))
            vectors[bad] = self._run(self._cpu(), [texts[i] for i in bad])
            bad = [i for i in bad if not np.isfinite(vectors[i]).all()]
        if bad:
            raise ValueError(f"The embedding of text {bad[0]} is not a number, even on the CPU.")
        return vectors.tolist()

    def encode(self, texts):
        try:
            return self._encode(texts)
        except RuntimeError as error:
            if self.device == "cpu":
                raise
            self._use_cpu(error)
            return self._encode(texts)

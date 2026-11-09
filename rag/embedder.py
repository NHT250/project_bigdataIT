"""all-MiniLM-L6-v2 embeddings on the GPU, or on the CPU when the GPU fails."""
import logging

import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

log = logging.getLogger(__name__)


class Embedder:
    def __init__(self):
        self._load("cuda" if torch.cuda.is_available() else "cpu", fall_back=True)

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

    def _encode(self, texts):
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def encode(self, texts):
        try:
            return self._encode(texts)
        except RuntimeError as error:
            if self.device == "cpu":
                raise
            self._use_cpu(error)
            return self._encode(texts)

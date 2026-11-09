"""The embedder keeps working on the CPU when the GPU fails."""
import numpy as np
import pytest
import torch

from rag import embedder as embedder_module
from rag.embedder import Embedder

CUDA_ERROR = torch.AcceleratorError("CUDA error: out of memory")


class FakeModels:
    """Stands in for SentenceTransformer and records which devices were loaded."""

    def __init__(self, monkeypatch, gpu=True, load_fails_on=(), encode_fails_on=()):
        self.devices = []
        self.load_fails_on = load_fails_on
        self.encode_fails_on = encode_fails_on
        monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
        monkeypatch.setattr(embedder_module, "SentenceTransformer", self.load)

    def load(self, name, device):
        self.devices.append(device)
        if device in self.load_fails_on:
            raise CUDA_ERROR
        return FakeModel(device, self.encode_fails_on)


class FakeModel:
    def __init__(self, device, encode_fails_on):
        self.device = device
        self.encode_fails_on = encode_fails_on
        self.max_seq_length = 256
        self.tokenizer = f"tokenizer on {device}"

    def encode(self, texts, **kwargs):
        if self.device in self.encode_fails_on:
            raise CUDA_ERROR
        return np.array([[1.0, 0.0] for _ in texts])


def test_healthy_gpu_is_used(monkeypatch):
    models = FakeModels(monkeypatch)
    assert Embedder().encode(["a"]) == [[1.0, 0.0]]
    assert models.devices == ["cuda"]


def test_machine_without_gpu_uses_the_cpu(monkeypatch):
    models = FakeModels(monkeypatch, gpu=False)
    Embedder().encode(["a"])
    assert models.devices == ["cpu"]


def test_gpu_failure_while_loading_falls_back_to_the_cpu(monkeypatch):
    models = FakeModels(monkeypatch, load_fails_on=("cuda",))
    assert Embedder().encode(["a"]) == [[1.0, 0.0]]
    assert models.devices == ["cuda", "cpu"]


def test_gpu_failure_while_encoding_falls_back_to_the_cpu(monkeypatch):
    models = FakeModels(monkeypatch, encode_fails_on=("cuda",))
    assert Embedder().encode(["a"]) == [[1.0, 0.0]]
    assert models.devices == ["cuda", "cpu"]


def test_after_falling_back_the_gpu_is_not_tried_again(monkeypatch):
    models = FakeModels(monkeypatch, encode_fails_on=("cuda",))
    embedder = Embedder()
    embedder.encode(["a"])
    embedder.encode(["b"])
    assert models.devices == ["cuda", "cpu"]


def test_cpu_model_keeps_the_long_sequence_limit_and_its_tokenizer(monkeypatch):
    FakeModels(monkeypatch, load_fails_on=("cuda",))
    embedder = Embedder()
    assert embedder.model.max_seq_length == 512
    assert embedder.tokenizer == "tokenizer on cpu"


def test_failure_on_the_cpu_is_not_hidden(monkeypatch):
    FakeModels(monkeypatch, gpu=False, encode_fails_on=("cpu",))
    with pytest.raises(RuntimeError, match="out of memory"):
        Embedder().encode(["a"])


def test_load_failure_on_the_cpu_is_not_hidden(monkeypatch):
    FakeModels(monkeypatch, load_fails_on=("cuda", "cpu"))
    with pytest.raises(RuntimeError, match="out of memory"):
        Embedder()

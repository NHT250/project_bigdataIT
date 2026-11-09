"""Import this before unsloth.

Unsloth's torch.compile of decoder blocks crashes with CUDA_ERROR_ILLEGAL_ADDRESS
on this Windows setup (torch 2.14.1+cu130, RTX 4050), so it runs eagerly instead.
"""
import os

os.environ.setdefault("UNSLOTH_BLOCK_COMPILE", "0")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

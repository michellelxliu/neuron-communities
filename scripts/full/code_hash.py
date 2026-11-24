#!/usr/bin/env python3
"""SHA-256 over neuron_communities/**/*.py (relative path + contents, sorted): identifies the harvest code without git."""
import hashlib
from pathlib import Path

root = Path(__file__).resolve().parents[2]
h = hashlib.sha256()
for f in sorted((root / "neuron_communities").rglob("*.py")):
    if "__pycache__" in f.parts:
        continue
    h.update(str(f.relative_to(root)).encode() + b"\0" + f.read_bytes() + b"\0")
print(h.hexdigest())

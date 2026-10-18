"""The SHA-256 of the files whose content defines a run; standard library only, so push.py needs no torch."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).parent
CODE_FILES = ["train.py", "provenance.py", "prepare_data.py", "attention_kernel.py", "hyperloop_kernel.py",
              "grt_kernel.py", "mesh_kernel.py", "requirements.txt", "tokenizer/tokenizer.json"]


def code_sha256() -> str:
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update(name.encode() + b"\0" + (ROOT / name).read_bytes())
    return digest.hexdigest()

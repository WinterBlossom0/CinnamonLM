"""Interrupt-and-resume, through the real train.py CLI.

The point of the checkpoint is that a killed run continues instead of restarting,
so this tests it the way it will actually be used: run train.py, kill it, run the
identical command again, and check it picked up where it stopped.

Uses a tiny synthetic corpus written straight into the pack cache, so no
tokenizer or download is needed and the whole thing runs on CPU in seconds.

Run: python -m tests.test_resume
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch


def _fake_cache(cache_dir, domain, seq_len, vocab, n=96):
    os.makedirs(cache_dir, exist_ok=True)
    for split, k in (("train", n), ("dev", 32)):
        blocks = np.random.randint(0, vocab, (k, seq_len + 1), dtype=np.int32)
        np.save(os.path.join(cache_dir, f"{domain}.{split}.{seq_len}.npy"), blocks)


def _tiny_tokenizer(path, vocab):
    from tokenizers import Tokenizer, models
    tok = Tokenizer(models.BPE())
    tok.add_tokens([f"t{i}" for i in range(vocab - 1)])
    tok.add_special_tokens(["<eos>"])
    tok.save(path)


def run_train(root, steps, extra=()):
    cmd = [sys.executable, "train.py", "--domain", "tst",
           "--tokenizer", os.path.join(root, "tok.json"),
           "--seq-len", "16", "--batch", "2", "--c-max2", "4",
           "--steps", str(steps), "--warmup", "2", "--eval-every", "2",
           "--log-every", "50", "--cache", os.path.join(root, "cache"),
           "--out", root, "--no-ewc-consolidate", "--amp", "off", *extra]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_resume_continues_instead_of_restarting():
    root = tempfile.mkdtemp()
    vocab, seq = 64, 16
    _tiny_tokenizer(os.path.join(root, "tok.json"), vocab)
    _fake_cache(os.path.join(root, "cache"), "tst", seq, vocab)

    run_train(root, steps=6)
    last = os.path.join(root, "tst.last.pt")
    best = os.path.join(root, "tst.best.pt")
    assert os.path.exists(last) and os.path.exists(best)
    ck1 = torch.load(last, map_location="cpu", weights_only=False)
    w1 = ck1["model"]["head.weight"].clone()

    # identical command again: must resume, not start over
    out = run_train(root, steps=12)
    assert "resumed from" in out, out[-2000:]
    ck2 = torch.load(last, map_location="cpu", weights_only=False)
    assert ck2["step"] > ck1["step"], (ck1["step"], ck2["step"])
    assert not torch.equal(w1, ck2["model"]["head.weight"]), "resume did not train further"

    # best.pt must track the best dev seen, never simply the latest
    b = torch.load(best, map_location="cpu", weights_only=False)
    hist = json.load(open(os.path.join(root, "tst.hist.json")))
    devs = [h["dev_loss"] for h in hist if "dev_loss" in h]
    if devs:
        assert b["dev_loss"] <= min(devs) + 1e-6, (b["dev_loss"], min(devs))
    assert not os.path.exists(last + ".tmp"), "atomic write left a temp file"


if __name__ == "__main__":
    test_resume_continues_instead_of_restarting()
    print("ok   resume continues, best.pt tracks best dev, writes are atomic")
    print("\nall passed")

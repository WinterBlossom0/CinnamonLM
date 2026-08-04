"""Dataset registry and BabyLM 2026 loading / packing.

The eight domains the assembled model will eventually use.  Phase 1 trains one
expert at a time, but the tokenizer is trained on all eight so it never has to be
rebuilt when a later expert arrives.
"""
import os
import sys

# name -> (hf repo, config, how to turn a row into text)
DATASETS = {
    "babylm":      ("BabyLM-community/BabyLM-2026-Strict", None, lambda r: r["text"]),
    "wikitext":    ("Salesforce/wikitext", "wikitext-103-raw-v1", lambda r: r["text"]),
    "amps_khan":   ("XinyaoHu/AMPS_khan", None,
                    lambda r: f'{r["problem"]}\n{r["hints/solutions"]}'),
    "xlam":        ("Salesforce/xlam-function-calling-60k", None,
                    lambda r: f'{r.get("query","")}\n{r.get("tools","")}\n{r.get("answers","")}'),
    "reasoning":   ("reasoning-core/procedural-warmup", None,
                    lambda r: f'{r["prompt"]}\n{r["answer"]}'),
    "code":        ("code-search-net/code_search_net", "all",
                    lambda r: r["whole_func_string"]),
    "wikitablet":  (None, None, None),      # local only, see LOCAL_FILES
    "textbooks":   ("izumi-lab/open-text-books", None, lambda r: r["text"]),
}

# Gated behind a licence click; the mirror is a byte-identical community copy.
FALLBACKS = {"xlam": ("NobodyExistsOnTheInternet/xlam-function-calling-60k", None)}

# WikiTableT (Chen et al. 2021) has no HF mirror -- the GitHub repo is code-only
# and the corpus is a 1.1 GB Google Drive zip.  extract_wikitablet.py pulls the
# natural text out of it (the release is pre-BPE'd, so the "@@ " continuation
# markers have to be stripped first) and writes the file these paths look for.
LOCAL_FILES = {
    "wikitablet": ["raw/wikitablet.txt", "/kaggle/input/**/wikitablet.txt"],
}


def local_path(name):
    """First existing local file for a dataset, or None.  Globs so it works
    wherever Kaggle decides to mount the attached dataset."""
    import glob
    for pat in LOCAL_FILES.get(name, []):
        hits = glob.glob(pat, recursive=True) if "*" in pat else (
            [pat] if os.path.exists(pat) else [])
        if hits:
            return hits[0]
    return None


def _file_iter(path, max_bytes, skip_bytes, chunk_lines=2000):
    sent = skipped = 0
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if skipped < skip_bytes:
                skipped += len(line)
                continue
            buf.append(line)
            if len(buf) >= chunk_lines:
                t = "".join(buf)
                sent += len(t)
                yield t
                buf = []
                if sent >= max_bytes:
                    return
        if buf:
            yield "".join(buf)

BABYLM_TRAIN = "BabyLM-community/BabyLM-2026-Strict"
BABYLM_DEV = "BabyLM-community/BabyLM-dev"
BABYLM_DOMAINS = ["bnc_spoken", "childes", "gutenberg",
                  "open_subtitles", "simple_wiki", "switchboard"]


def corpus_iter(name, max_bytes=80 * 2**20, log=print, split="train", skip_bytes=0):
    """Stream up to max_bytes of text from one registry entry.

    Streaming so a 4 GB corpus never lands on disk just to sample 80 MB of it.
    A dataset that cannot be reached is skipped loudly rather than killing a job
    that is otherwise fine.

    skip_bytes discards a prefix first, which is how a held-out slice is carved
    from a corpus that ships no validation split of its own: the training run
    consumes the first N bytes, validation starts after them, so the two never
    share a document.
    """
    from datasets import load_dataset

    repo, cfg, to_text = DATASETS[name]
    path = local_path(name)
    if path:
        log(f"  + {name}[{split}]: local {path}")
        yield from _file_iter(path, max_bytes, skip_bytes)
        return
    if repo is None:
        log(f"  X {name}: local-only dataset, file not found "
            f"(expected one of {LOCAL_FILES.get(name)}) -- SKIPPED")
        return
    for attempt, (r, c) in enumerate([(repo, cfg)] + [FALLBACKS.get(name)] * (name in FALLBACKS)):
        if r is None:
            continue
        try:
            ds = load_dataset(r, c, split=split, streaming=True)
            if attempt:
                log(f"  ! {name}: {repo} unreachable, using mirror {r}")
            sent = skipped = 0
            for row in ds:
                try:
                    t = to_text(row)
                except KeyError:                       # mirror with a different schema
                    t = next((v for v in row.values() if isinstance(v, str)), "")
                if not t or not t.strip():
                    continue
                if skipped < skip_bytes:
                    skipped += len(t)
                    continue
                sent += len(t)
                yield t
                if sent >= max_bytes:
                    log(f"  + {name}[{split}]: {sent/2**20:.0f} MB")
                    return
            log(f"  + {name}[{split}]: {sent/2**20:.0f} MB (exhausted)")
            return
        except Exception as e:
            log(f"  ! {name}[{split}]: {r} failed -- {type(e).__name__}: {str(e)[:120]}")
    log(f"  X {name}: SKIPPED, no reachable source")


def dev_iter(name, train_bytes, max_bytes=8 * 2**20, log=print):
    """Held-out text for a non-BabyLM domain.

    Prefer the dataset's own validation/test split.  Only if it has neither fall
    back to the tail of train, past everything the training run consumed --
    re-reading train from the start would put the same documents on both sides.
    """
    from datasets import get_dataset_split_names

    repo, cfg, _ = DATASETS[name]
    if repo is None or local_path(name):        # local file: hold out the tail
        yield from corpus_iter(name, max_bytes, log, skip_bytes=train_bytes)
        return
    try:
        avail = get_dataset_split_names(repo, cfg)
    except Exception:
        avail = []
    for s in ("validation", "valid", "test"):
        if s in avail:
            yield from corpus_iter(name, max_bytes, log, split=s)
            return
    log(f"  ~ {name}: no val/test split, holding out the tail past {train_bytes/2**20:.0f} MB")
    yield from corpus_iter(name, max_bytes, log, split="train", skip_bytes=train_bytes)


def all_corpora(max_bytes_each=80 * 2**20, log=print):
    for name in DATASETS:
        log(f"[{name}]")
        yield from corpus_iter(name, max_bytes_each, log)


# --------------------------------------------------------------------------- #
# BabyLM 2026
# --------------------------------------------------------------------------- #

def babylm_files(split):
    """Download the raw per-domain files.

    Train and validation come from two *different repos*, both released by the
    organisers: BabyLM-2026-Strict for train, BabyLM-dev for validation.  Do not
    substitute a random split of the training file -- these corpora are ordered
    documents (CHILDES transcripts, Gutenberg books), so consecutive lines are
    from the same document and a random line split leaks it across both sides,
    making validation loss optimistic.  BabyLM-Test stays untouched.
    """
    from huggingface_hub import hf_hub_download

    repo, suffix = ((BABYLM_TRAIN, ".train.txt") if split == "train"
                    else (BABYLM_DEV, ".dev"))
    return {d: hf_hub_download(repo, d + suffix, repo_type="dataset")
            for d in BABYLM_DOMAINS}


def babylm_text(split, log=print):
    """Yield text per domain, in file order so document context survives."""
    for domain, path in babylm_files(split).items():
        with open(path, encoding="utf-8") as f:
            buf = []
            for line in f:
                buf.append(line)
                if len(buf) >= 10_000:
                    yield "".join(buf)
                    buf = []
            if buf:
                yield "".join(buf)
        log(f"  read {split}/{domain}")


def pack(texts, tok, seq_len, eos_id, log=print, log_every=50, batch=64):
    """Tokenise a stream and pack into contiguous [n, seq_len+1] blocks.

    seq_len+1 because the training step needs one extra token to shift labels
    against.  Nothing overlaps; the tail that does not fill a block is dropped,
    which at these corpus sizes is a rounding error.

    encode_batch rather than encode: the Rust tokenizer releases the GIL and
    threads across cores, which turns ~30 minutes of packing BabyLM into a few.
    """
    import numpy as np

    dtype = np.uint16 if tok.get_vocab_size() < 65536 else np.int32
    out, carry, n, seen = [], [], 0, 0

    def flush(chunk):
        nonlocal carry, n
        for enc in tok.encode_batch(chunk):
            carry.extend(enc.ids)
            carry.append(eos_id)
            while len(carry) >= seq_len + 1:
                out.append(np.array(carry[:seq_len + 1], dtype=dtype))
                carry = carry[seq_len:]      # next block starts at the label token
                n += 1

    buf = []
    for t in texts:
        buf.append(t)
        if len(buf) >= batch:
            flush(buf)
            seen += 1
            buf = []
            if log_every and seen % log_every == 0:
                log(f"    {n} blocks ({n*seq_len/1e6:.1f}M tokens)")
    if buf:
        flush(buf)
    log(f"    {n} blocks ({n*seq_len/1e6:.1f}M tokens) total")
    return np.stack(out) if out else np.zeros((0, seq_len + 1), dtype=dtype)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "probe"
    if which == "probe":                     # cheap reachability check, no big downloads
        for name in DATASETS:
            print(f"[{name}]")
            it = corpus_iter(name, max_bytes=2000)
            got = sum(1 for _ in it)
            print(f"  -> {got} samples")
    elif which == "babylm":
        for split in ("train", "dev"):
            files = babylm_files(split)
            tot = sum(os.path.getsize(p) for p in files.values())
            print(f"{split}: {len(files)} domains, {tot/2**20:.1f} MB")

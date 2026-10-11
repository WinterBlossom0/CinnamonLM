"""Build the 8-gram index of benchmark evaluation splits for decontam.py. Keeps hashes and item metadata, never benchmark text.

Usage: decontam_index.py (with the data_pipeline venv)
Writes reports/decontam/eval_ngrams.npy (sorted uint64 8-gram hashes), eval_ngram_items.npy (item index of each hash)
and eval_items.json (benchmark, split, example index, kind, distinct 8-gram count).
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.common.dedup import MinHasher, MinHashParams, dedup_words  # noqa: E402
from workflow.corpus import REPORTS  # noqa: E402

OUT = REPORTS / "decontam"

# PaLM (arXiv:2204.02311, §8) measures contamination with 8-grams.
NGRAM = 8
hasher = MinHasher(MinHashParams(ngram_size=NGRAM))
# The Hugging Face copy of SIQA needs a loading script that `datasets` no longer runs; this is its official source.
SIQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
SIQA_ZIP = Path.home() / ".cache/socialiqa/socialiqa-train-dev.zip"
SIQA_SHA256 = "ee073914a0fc33265cfbcfc50ec20df9b2e07809c3f420f599138d1f394ef5c3"
# Pinned Hugging Face revisions, so the index is the same on every build.
REVISIONS = {
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "allenai/winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "baber/piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
    "aps/super_glue": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
    "cais/mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "Idavidrein/gpqa": "83022cefff930aea54f654c0b282e74b9eeda5c6",
    "EleutherAI/lambada_openai": "900124bf3b8235c6daf21033af9948b3f07346c4",
}


def pinned(name, *args, **kwargs):
    return load_dataset(name, *args, revision=REVISIONS[name], **kwargs)


def gold_choice(choices, key):
    return choices["text"][choices["label"].index(key)] if key in choices["label"] else ""


def qa_items():
    """(benchmark, split, index, text) for question-answer benchmarks: input plus gold answer (input only when unlabeled)."""
    for split in ("validation", "test"):
        for index, row in enumerate(pinned("Rowan/hellaswag", split=split)):
            label = row["label"]
            ending = row["endings"][int(label)] if label not in ("", None) else ""
            yield "hellaswag", split, index, f"{row['ctx']} {ending}"
        for config in ("ARC-Easy", "ARC-Challenge"):
            for index, row in enumerate(pinned("allenai/ai2_arc", config, split=split)):
                yield config.lower(), split, index, f"{row['question']} {gold_choice(row['choices'], row['answerKey'])}"
        for index, row in enumerate(pinned("allenai/winogrande", "winogrande_xl", split=split)):
            option = {"1": row["option1"], "2": row["option2"]}.get(row["answer"], "")
            yield "winogrande", split, index, row["sentence"].replace("_", option) if option else row["sentence"]
        for index, row in enumerate(pinned("baber/piqa", split=split)):
            solution = {0: row["sol1"], 1: row["sol2"]}.get(row["label"], "")
            yield "piqa", split, index, f"{row['goal']} {solution}"
        for index, row in enumerate(pinned("aps/super_glue", "boolq", split=split)):
            yield "boolq", split, index, row["question"]
        for index, row in enumerate(pinned("cais/mmlu", "all", split=split)):
            yield "mmlu", split, index, f"{row['question']} {row['choices'][row['answer']]}"
    for index, (row, label) in enumerate(siqa_dev()):
        yield "siqa", "validation", index, f"{row['context']} {row['question']} {row['answer' + 'ABC'[int(label) - 1]]}"
    for index, row in enumerate(pinned("openai/gsm8k", "main", split="test")):
        yield "gsm8k", "test", index, f"{row['question']} {row['answer']}"
    for config in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                   "prealgebra", "precalculus"):
        for index, row in enumerate(pinned("EleutherAI/hendrycks_math", config, split="test")):
            yield f"math_{config}", "test", index, f"{row['problem']} {row['solution']}"
    for index, row in enumerate(pinned("Idavidrein/gpqa", "gpqa_main", split="train")):
        yield "gpqa", "train", index, f"{row['Question']} {row['Correct Answer']}"


def siqa_dev():
    """(row, label) pairs of the SIQA dev split; the test labels are not public."""
    if not SIQA_ZIP.exists():
        SIQA_ZIP.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SIQA_URL, SIQA_ZIP)
    if hashlib.sha256(SIQA_ZIP.read_bytes()).hexdigest() != SIQA_SHA256:
        raise ValueError(f"{SIQA_ZIP} is not the pinned SIQA release")
    with zipfile.ZipFile(SIQA_ZIP) as archive:
        rows = archive.read("socialiqa-train-dev/dev.jsonl").decode("utf-8").splitlines()
        labels = archive.read("socialiqa-train-dev/dev-labels.lst").decode("utf-8").split()
    if len(rows) != len(labels):
        raise ValueError(f"SIQA dev: {len(rows)} rows but {len(labels)} labels")
    return [(json.loads(row), label) for row, label in zip(rows, labels)]


def passage_items():
    """(benchmark, split, index, text) for language-modelling benchmarks whose evaluation text is a passage."""
    for index, row in enumerate(pinned("EleutherAI/lambada_openai", "default", split="test")):
        yield "lambada", "test", index, row["text"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items, hash_pieces, item_pieces = [], [], []
    skipped_short = {}

    def add(benchmark, split, index, text, kind):
        words = dedup_words(text)
        if len(words) < NGRAM:
            skipped_short[benchmark] = skipped_short.get(benchmark, 0) + 1
            return
        hashes = hasher.shingles(words)
        item_pieces.append(np.full(len(hashes), len(items), dtype=np.int32))
        hash_pieces.append(hashes)
        items.append({"benchmark": benchmark, "split": split, "index": index, "kind": kind,
                      "ngrams": int(len(hashes))})

    for benchmark, split, index, text in qa_items():
        add(benchmark, split, index, text, "qa")
    for benchmark, split, index, text in passage_items():
        add(benchmark, split, index, text, "passage")
    hashes = np.concatenate(hash_pieces)
    item_ids = np.concatenate(item_pieces)
    order = np.argsort(hashes, kind="stable")
    np.save(OUT / "eval_ngrams.npy", hashes[order])
    np.save(OUT / "eval_ngram_items.npy", item_ids[order])
    counts = {}
    for item in items:
        key = f"{item['benchmark']}/{item['split']}"
        counts[key] = counts.get(key, 0) + 1
    (OUT / "eval_items.json").write_text(json.dumps({"ngram": NGRAM, "revisions": REVISIONS, "siqa_sha256": SIQA_SHA256,
                                                     "items": items, "items_per_split": counts,
                                                     "skipped_under_8_words": skipped_short}))
    print(json.dumps(counts, indent=1))
    print("skipped (< 8 words):", skipped_short, "| items:", len(items), "| 8-gram entries:", len(hashes))


if __name__ == "__main__":
    main()

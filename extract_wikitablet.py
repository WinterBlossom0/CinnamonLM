"""Extract natural text from the real WikiTableT release (Chen et al. 2021).

    https://github.com/mingdachen/WikiTableT  ->  final_data.zip (Google Drive)

Two things the raw release needs before it is usable as tokenizer training text:

  1. Every field is already BPE-tokenised with 30k merges, marked by "@@ "
     continuations ("Au@@ tism", "child@@ 's", "four-@@ fif@@ ths").  Training our
     tokenizer on that would teach it someone else's subword fragments, so the
     markers are removed to recover the original text.
  2. train.json is 1.8 GB; the tokenizer only needs a sample, so this streams
     straight out of the zip and stops at --mb.

    python extract_wikitablet.py --zip raw/wikitablet_final_data.zip --mb 150
"""
import argparse
import json
import os
import zipfile


def debpe(s):
    """Undo subword-nmt style continuation markers."""
    return s.replace("@@ ", "")


def record_text(r):
    """Title, section headings, data tuples and prose -- the table-to-text pair."""
    parts = [r.get("doc_title", "")]
    sec = r.get("sec_title") or []
    parts += sec if isinstance(sec, list) else [sec]
    for tup in (r.get("data") or []):
        parts.append(" : ".join(tup) if isinstance(tup, list) else str(tup))
    parts.append(r.get("text", ""))
    return debpe("\n".join(p for p in parts if p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="raw/wikitablet_final_data.zip")
    ap.add_argument("--member", default="final_data/train.json")
    ap.add_argument("--mb", type=int, default=150)
    ap.add_argument("--out", default="raw/wikitablet.txt")
    a = ap.parse_args()

    limit, written, n = a.mb * 2**20, 0, 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with zipfile.ZipFile(a.zip) as z, z.open(a.member) as f, \
            open(a.out, "w", encoding="utf-8") as out:
        for line in f:
            try:
                t = record_text(json.loads(line.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not t.strip():
                continue
            out.write(t + "\n\n")
            written += len(t) + 2
            n += 1
            if written >= limit:
                break
    print(f"wrote {n} records, {written/2**20:.1f} MB -> {a.out}")
    with open(a.out, encoding="utf-8") as f:
        head = f.read(600)
    assert "@@ " not in head, "de-BPE failed"
    print("--- sample ---")
    print(head[:500])


if __name__ == "__main__":
    main()

"""Train the 128k byte-level BPE on all eight corpora.

Runs unchanged locally or as a Kaggle kernel (it just writes to /kaggle/working
when that exists).

Three rules are required of this tokenizer:
  1. special characters stay separate tokens
  2. numbers stay separate tokens
  3. neither may merge with anything else, including each other

These are enforced *structurally*, not by post-filtering the vocabulary: BPE can
only ever merge inside a pre-token, never across a pre-token boundary.  So if the
pre-tokenizer emits every digit and every non-letter symbol as its own piece, no
merge crossing those boundaries can exist in the first place.

Pre-tokenizer order matters, and was picked by measurement, not taste:

  Digits and the symbol split run on RAW text, ByteLevel runs last with
  add_prefix_space=False.

  ByteLevel first would fragment non-ASCII: "naive" with a diaeresis encodes to
  bytes whose continuation byte lands on U+00AF, a Unicode symbol, so the symbol
  rule would fire *inside* a character and split it ("naA", "-", "ve").
  add_prefix_space=True would prefix every pre-token with the space marker,
  producing "G4", "G$", "G!" -- precisely the merge rule 3 forbids.
  Running on raw text sidesteps both: digits are digits, symbols are symbols, and
  accented letters stay whole and mergeable.

Consequence, accepted: a number preceded by a space costs a standalone space
token, because attaching it would merge a number with something else.  The very
first token of a document also loses its leading space, as in GPT-2.
"""
import argparse
import os
import sys
import time
import unicodedata

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ["<pad>", "<bos>", "<eos>"]
SYMBOL_RE = r"[^\p{L}\p{N}\s]"


def build():
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.Split(Regex(SYMBOL_RE), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def _is_number(ch):
    """Mirror Rust's char::is_numeric, which is what the Digits pre-tokenizer uses.

    Not str.isnumeric(): Python also returns True for CJK ideographs with numeric
    values, so a normal word like 一致 would read as a merged number even though
    the pre-tokenizer (correctly) treats 一 as the letter it is categorised as.
    """
    return unicodedata.category(ch) in ("Nd", "Nl", "No")


def _is_symbol(ch):
    """Complement of [^\\p{L}\\p{N}\\s], the class the symbol split isolates."""
    return not (unicodedata.category(ch)[0] in "LN" or ch.isspace())


def check_rules(tok, verbose=True):
    """Check the three rules against the trained vocabulary itself.

    Vocabulary pieces are byte-level encoded, so they must be decoded back to real
    text before testing -- otherwise a UTF-8 continuation byte reads as a symbol
    and legitimate merges look illegal.
    """
    dec = decoders.ByteLevel()
    bad = []
    for piece in tok.get_vocab():
        if piece in SPECIALS:
            continue
        raw = dec.decode([piece])
        if len(raw) < 2 or "�" in raw:    # single char, or a byte fragment
            continue
        for ch in raw:
            if _is_number(ch):
                bad.append(("number merged", piece, raw))
                break
            if _is_symbol(ch):
                bad.append(("symbol merged", piece, raw))
                break
    assert not bad, f"{len(bad)} illegal merges, e.g. {bad[:10]}"

    probe = 'Order 42 items for $1,250.99 -- foo_bar(x=3); "ok"!!!\nnaïve 2026'
    ids = tok.encode(probe).ids
    assert tok.decode(ids) == probe, f"round-trip failed:\n{probe!r}\n{tok.decode(ids)!r}"
    pieces = [tok.id_to_token(i) for i in ids]
    if verbose:
        print(f"  vocab {tok.get_vocab_size()}, probe -> {len(ids)} tokens")
    return pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=128_000)
    ap.add_argument("--mb-per-source", type=int, default=80)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = a.out or ("/kaggle/working/tokenizer.json"
                    if os.path.isdir("/kaggle/working") else "tokenizer.json")
    from cinnamon.data import all_corpora

    tok = build()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # all 256 bytes -> no unk
        show_progress=True,
    )

    t0 = time.time()
    print(f"training {a.vocab} BPE on 8 corpora, {a.mb_per_source} MB each", flush=True)
    tok.train_from_iterator(all_corpora(a.mb_per_source * 2**20), trainer=trainer)
    print(f"trained in {(time.time()-t0)/60:.1f} min, vocab {tok.get_vocab_size()}", flush=True)

    # Save before validating: an over-strict check must never destroy half an
    # hour of training.  Verify the artefact, do not gate its existence.
    tok.save(out)
    print("saved", out, f"{os.path.getsize(out)/2**20:.1f} MB", flush=True)
    pieces = check_rules(tok)
    print("rule check passed. sample:", pieces[:40], flush=True)


if __name__ == "__main__":
    main()

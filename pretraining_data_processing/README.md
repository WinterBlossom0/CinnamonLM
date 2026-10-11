# Commercial-use pretraining corpus from Common Pile v0.1

Seven Common Pile v0.1 sources, filtered to text that permits commercial use, cleaned per source, English only,
decontaminated against evaluation benchmarks and deduplicated across sources. One script rebuilds the corpus from
pinned Hugging Face revisions; every stage records its input and output shard hashes.

Dataset: [WinterBlossom/cinnamon-pretraining-corpus](https://huggingface.co/datasets/WinterBlossom/cinnamon-pretraining-corpus)

## Pipeline

| Stage | What it does |
|---|---|
| Download | Pinned revision of each `common-pile/<source>` repo; every shard checked against its LFS sha256 (`workflow/raw_manifest.json`) |
| Licence | Drops documents whose own text restricts the work (NC/ND, all rights reserved, "used with permission", third-party lyrics); strips restricted media credits. An inline statement overrides the metadata licence ([rules](licence/REPORT.md)) |
| Clean | Source-specific pipeline (`data_pipeline/sources/<source>.py`): markup and boilerplate removal, OCR repair, English filter (fastText LID), exact and MinHash near dedup inside the source, PII masking |
| Decontam | Removes evaluation items: a match shares ≥ 8 and ≥ 70 % of the item's or the document's 8-grams (PaLM threshold); matched lines are removed, documents that are ≥ 50 % matched are dropped |
| Dedup | Drops a document when ≥ 80 % of its 13-grams occur in a source of higher priority (Common Pile / Dolma BFF, DCLM parameters, Bloom filter FPR 0.6 %) |
| Publish | Moves the build into `data/common_pile/<source>/` |

Decontaminated benchmarks: LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, SIQA, BoolQ, MMLU, GSM8K,
MATH, GPQA (validation and test splits, pinned revisions).

Dedup priority: Public Domain Review, Project Gutenberg, LibreTexts, PressBooks, OER Commons, DOAB, Wikimedia. Whole
Gutenberg books win over OER anthology chapters that reproduce them.

## Result

| Source | Raw docs | After licence | After clean | After decontam | Final docs | Final chars |
|---|---:|---:|---:|---:|---:|---:|
| Public Domain Review | 1,412 | 1,412 | 1,358 | 1,358 | 1,358 | 6.9 M |
| Project Gutenberg | 71,810 | 71,222 | 52,124 | 52,124 | 52,124 | 19.20 B |
| LibreTexts | 62,269 | 61,238 | 39,624 | 39,623 | 39,352 | 312.2 M |
| PressBooks | 106,881 | 103,850 | 56,510 | 56,507 | 45,880 | 430.6 M |
| OER Commons | 9,339 | 9,036 | 4,912 | 4,912 | 4,047 | 28.0 M |
| DOAB | 474,445 | 474,102 | 358,349 | 358,349 | 358,345 | 10.32 B |
| Wikimedia | 63,969,938 | 63,957,575 | 6,758,048 | 6,758,047 | 6,715,124 | 26.65 B |
| **Total** | **64,696,094** | **64,678,435** | **7,270,925** | **7,270,920** | **7,216,230** | **56.94 B** |

Decontamination also removed 538 lines from 188 documents (mostly MMLU passages quoted in history and science texts).

## Reproducibility

- `workflow/build_corpus.py` runs every stage in order from the pinned download; `data_pipeline/uv.lock` pins the Python environment.
- Each stage checks its input against the previous stage record and writes `reports/<source>/build/<stage>.json`
  (shard sha256 in and out, counts, code sha256). Decontam rows: `reports/decontam/`; dedup decisions: `reports/cross_dedup/`.
- Two separate builds of Public Domain Review and OER Commons gave identical shard sha256 for download, licence and clean.
- Records keep the code sha256 at run time. `workflow/corpus.py` and `workflow/build_corpus.py` changed during the build
  (dedup priority order, download and resume handling, gzip header fix) without a change to stage logic; the final
  shards were rewritten with the fixed gzip writer, with decompressed content unchanged (`output_content_sha256` in `dedup.json`).

## Layout

| Path | Content |
|---|---|
| `workflow/` | Build driver, decontamination, cross-source dedup, raw manifest |
| `licence/` | Licence filter, its apply step and rules; the test fixtures are not distributed (they hold non-commercial records) |
| `data_pipeline/` | Per-source cleaning pipelines and tests |
| `reports/` | Stage records of the build, calibration evidence for thresholds |
| `analysis/` | Literature notes, per-source findings that motivated the rules, audit limits |

## Limits

- Overlaps below 80 % between sources stay (quotations, adaptations); a high-priority document contained in a longer
  lower-priority one keeps both copies; Wikimedia sub-projects are deduplicated only by in-source MinHash (Jaccard 0.8).
- Evaluation items shorter than 8 words cannot be matched.
- Known residues per source: `analysis/audits/stated_limits_previous_build.json`.

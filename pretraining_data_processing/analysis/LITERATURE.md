# Literature: preprocessing Common Pile v0.1 sources for pretraining

Sources in scope (raw versions on Hugging Face, `common-pile/<name>`): doab, libretexts, pressbooks,
oercommons, project_gutenberg, public_domain_review, wikimedia.

Reading protocol: full-text PDFs downloaded from arXiv (and ACL Anthology for two papers), extracted with
`pdftotext`, stored in
`/tmp/claude-1000/-home-vihaa-CinnamonLM/613996b4-284e-41da-b92e-c61d5552ebe8/scratchpad/common_pile_samples/papers/`.
Each entry lists the parts read. Where a paper is mostly about model evaluation (Gopher, T5, StarCoder,
MADLAD-400), only the data sections and data appendices were read, and the entry says so.
The Common Pile GitHub repository (`r-three/common-pile`, `filtering/` and `sources/`) was also read,
because the paper does not give units or the exact regexes. It is code, not a paper; it is marked "[repo]".

The entries are grouped by cause (a)–(i). Each paper appears under the cause it informs most. Its
contributions to other causes are cross-referenced.

---

## 0. What the Common Pile authors already did (paper 2506.05209 + [repo])

**Kandpal, Lester, Raffel, et al. 2025. "The Common Pile v0.1: An 8TB Dataset of Public Domain and Openly
Licensed Text." arXiv:2506.05209.** Read: full paper including Appendices B, C, F, J (Tables 5, 6), K–O.

Shared pipeline (§4.1, App. J), built on the Dolma toolkit, applied separately to each source and then globally:
- fastText language ID: keep documents with English score > 0.5.
- Unigram log-likelihood filter for "pervasive OCR errors" (peS2o method; Trillion Word Corpus unigrams).
  [repo] `perplexity_tagger.py`: score = mean log2 unigram probability per word (blingfire words, lowercased;
  unknown words get the probability of a fictitious count sqrt(V)+1). Remove documents with score ≤ −20.
- Two fastText toxicity classifiers (Jigsaw "hate" and "nsfw"). [repo] They are applied as **sentence-span
  deletion** with min_score 0.1 (the span is replaced by the empty string), not as document removal.
- Regex PII: e-mail, phone number, IP address replaced by `<EMAIL_ADDRESS>`, `<PHONE_NUMBER>`, `<IP_ADDRESS>`.
- Source-specific line removal ("regex filter" column of Table 5). [repo] `line_tagger.py` removes a
  paragraph only if its stripped text **exactly equals** one entry of a per-source list.
- Global fuzzy dedup: Dolma Bloom-filter (BFF) dedup, duplicate if > 90 % of 20-grams are already in the
  filter. [repo] `global_dedupe.json`: `ngram_length 20, stride 1, overlap_threshold 0.9`, paragraph
  separator set to a string that does not occur in text (so the unit is the whole document), Bloom filter
  false-positive rate 1e-2. This is a document-vs-corpus test that depends on processing order (DCLM App. L).
- No benchmark decontamination of these sources. The only contamination step in the paper: Winogrande was
  dropped from the evaluation because the DPI source contains it (§4.3).
- Doc-length thresholds are in **characters** ([repo] `char_length_v1`).

Per source (Table 5, Table 6, Table 7, Table 8, App. B; [repo] mixer configs):

| Source | Collection / extraction (App. B) | Filters (Table 5, [repo]) | Raw → filtered docs (GB) | Mix repeats (1T run) | Cool-down |
|---|---|---|---|---|---|
| DOAB | DOAB metadata feed; English, CC BY / BY-SA; PDFs → text with **Marker**; kept only books whose front/back matter contains a whitelisted license statement | lang > 0.5; length > 200 chars; toxicity span deletion > 0.1; PII; no line regex | 474,445 → 403,992 (12.5 → 12) | 6 | ×2, 16.04 % of cool-down |
| LibreTexts | per-section license check (PD, CC BY, BY-SA, GFDL); text from HTML pages | lang; length > 700; toxicity; PII | 62,269 → 40,049 (5.3 → 3.6) | 6 | ×2 |
| PressBooks | search query: English, PD/CC BY/BY-SA; web version of each book | lang; length > 600; toxicity; PII | 106,881 → 54,455 (1.5 → 0.6) | 6 | ×2 |
| OERCommons | search query: English, PD/CC BY/BY-SA; text from HTML pages | lang; length > 300; toxicity; PII | 9,339 → 5,249 (0.1 → 0.05) | 6 | ×2 |
| Project Gutenberg | English + PD per metadata, plus PG-19 books; "minimal preprocessing" to strip PG header/footer ([repo] `preprocess.py`: regex `START OF (THE|THIS) PROJECT GUTENBERG` / `END OF ...`); producer preambles **not** removed (stated in code docstring) | lang > 0.5; log-likelihood > −20; line list of 78 exact lines (`[Illustration]`, `*  *  *` separators, Distributed Proofreaders credit lines, ebookmaker WARNING lines); no PII; no length filter; no toxicity deletion | 71,810 → 55,454 (26.2 → 20.1) | 1 | none |
| Public Domain Review | all CC BY-SA articles | length > 100 chars and > 20 whitespace tokens; PII; line list with one line: "The text of this essay is published under a CC BY-SA license, see here for details." | 1,412 → 1,406 | 6 | ×2 |
| Wikimedia | March 2025 dumps of 8 English wikis (Wikipedia, Wikinews, Wikibooks, Wikiquote, Wikisource, Wikiversity, Wikivoyage, Wiktionary); latest revision; **includes talk pages**; wikitext math → LaTeX by custom code; wtf_wikipedia; remaining HTML tags removed by regex | lang > 0.5; length > 100 chars and > 20 tokens; drop titles starting "User talk"; line list of 1,554 exact lines (talk-page and bot boilerplate); PII; no toxicity deletion | 63,969,938 → 16,311,574 (90.5 → 57.4) | 6 | ×0.4 |

Other facts used:
- Mixing (§4.2, App. K): per-source 1.7B/28B-token models ranked sources; heuristic weights; target ≤ 6
  repeats over 1T tokens; small sources were **assumed** high quality and set to 6 repeats. MixMin did not
  beat the heuristic mixture. Comma v0.1-2T repeated the 1T mixture twice (up to 16 passes for some sources).
  Weight decay 0.2 instead of 0.1 "possibly due to the large amount of repetition".
- Comma is weak on HellaSwag/PIQA/CSQA and strong on MMLU/ARC (Table 9); authors attribute this to missing
  domains (blogs, hobbies, sports).
- License caveats (§2.1, App. C.1): license laundering; image-badge licenses missed by string matching;
  quoted in-copyright snippets inside open documents.

---

## (a) Duplication: exact, near-duplicate, cross-source, repeated boilerplate

### Patterns (from the papers)
- Templated near-duplicates that differ only in names, numbers, places (Lee et al. Table 1; RefinedWeb H.1).
- Same text under different identifiers: PG editions with different PG IDs (Gerlach & Font-Clos);
  BookCorpus 2,930 of 7,185 books duplicated, plus **truncated** copies (30 % or first 28 words) and copies
  differing in 15–30 lines (Bandy & Vincent); repeated page scans (IB-ET: 18.4 % of volumes had duplicate pages).
- Repeated boilerplate inside documents that are otherwise unique: licenses, disclaimers, bylines,
  navigation; 70 % of CC paragraphs are duplicates (CCNet); 61-word sentence 61,036× in C4 (Lee et al.).
- Repetition clusters are heavy-tailed: in C4 1.8M near-dup clusters are pairs, 280 clusters > 5,000 docs
  (Lee et al.); 85 clusters > 1,000 copies in IB-HL-ET, mostly missed boilerplate and table filler.
- Outlier spikes in the document-length histogram reveal templates and truncation (WIMBD).
- Long documents give SimHash false positives (ROOTS: do not drop SimHash clusters of docs > 6,000 chars).

### Detection / fix methods with parameters
- MinHash-LSH document dedup: 5-grams, 9,000 hashes = 450 bands × 20 (Lee et al.; RefinedWeb), P(match | s=0.8) = 99.4 %;
  then verify edit similarity > 0.8 (Lee et al.). Cheaper: 112 hashes = 14 × 8, P = 56/77/92/98.8 % at
  s = 0.7/0.75/0.8/0.85 (FineWeb); 1,395 = 93 × 15 emulating 450 × 20 (DCLM). Gopher: 13-gram Jaccard > 0.8,
  normalized whitespace, punctuation ignored. Normalization before hashing: lowercase, remove punctuation,
  NFD + strip accents, normalize whitespace (RefinedWeb G.3.1).
- Exact substring (suffix array) removal: ≥ 50 BPE tokens (Lee et al.: the "knee" is at 10 tokens; 25-token
  matches had no false positives; doubled to 50). Removed 7.18 % of C4 tokens. Remove spans and drop docs
  with < 20 remaining chars (RefinedWeb EXACTSUBSTR-CUT); alternatives drop doc if > 20 % duplicated.
- Paragraph-hash dedup for boilerplate: normalize (lowercase, digits → 0, strip punctuation and accents),
  SHA-1 first 64 bits (CCNet). Dedup **before** LID because boilerplate in the wrong language misleads LID (CCNet).
- Per-site/per-collection repeated-line removal: drop lines that occur in > 1 % of pages of a domain (ROOTS).
- BFF paragraph+document Bloom dedup: min = max n-gram 13, threshold 0.8, FPR 0.01 (DCLM App. L);
  min n-gram 5 removes short list items (e.g. recipe lines) and lowered MMLU.
- Containment metric for partial copies: S_title = 2JC/(J+C), C = |N1∩N2| / min(|N1|,|N2|) (S2ORC; used there
  for titles, applies to truncated duplicates where Jaccard is low).
- Book-level SimHash: 7-char shingles; reject matches with different language or > 15 % difference in
  "continuous character count" (chars excluding whitespace, line breaks, hyphens); 97 % precision (IB 1.0).
- Page / paragraph SimHash for OCR books: 128-bit, 9-grams with ≥ 4 distinct chars, ≤ 6 bits (pages) /
  ≤ 5 bits (paragraphs), 6 bands, union-find (IB-ET).
- Semantic dedup (SemDeDup; D4) removes paraphrase-level duplicates; DCLM found SemDedup **hurt** in their
  setting (Table 4), so treat as optional.
- Global vs. per-shard dedup: global MinHash across 96 snapshots hurt; the 10 % kept from an old crawl was
  worse than the 90 % removed (FineWeb §3.4). Main gain is from removing the **large** clusters.

### Measured effects
- Gopher A.3.2: quality filter, then exact dedup, then fuzzy dedup each lowered loss (1.4B models).
- Lee et al.: dedup reduces verbatim emission ~10× (1.926 % → 0.189 %/0.138 % of tokens); same or lower ppl.
- Kandpal et al.: a sequence present 10× is generated ~1,000× more often than one present once.
- Carlini et al.: memorization grows log-linearly with duplicate count, model size, context length.
- Hernandez et al.: repeating 0.1 % of data 100× degrades an 800M model to ~400M-model loss; damage peaks
  in a middle range of repeat counts; copying/induction heads are hurt most.
- Muennighoff et al. App. (deduplicated C4): dedup did not change the optimal epoch count; in their
  downstream test, dedup filtering (100-char overlap) "does not help" while perplexity filtering helped.

### Papers
**Lee, Ippolito, Nystrom, Zhang, Eck, Callison-Burch, Carlini 2022. "Deduplicating Training Data Makes Language
Models Better." arXiv:2107.06499.** Read: full paper and appendices A–E. Used: NEARDUP (MinHash 5-grams,
b=20, r=450, Jaccard and edit-sim > 0.8) and EXACTSUBSTR (suffix array, k=50 tokens, threshold rationale);
duplicate rates (C4 3.04 % near-dup docs, Wiki-40B 0.39 %; 4.6 % of C4 validation has a near-dup in train);
caution that removing duplicated substrings can strip attribution text (§7, "text that gives attribution is
often duplicated").

**Kandpal, Wallace, Raffel 2022. "Deduplicating Training Data Mitigates Privacy Risks in Language Models."
arXiv:2202.06539.** Read: §1–2 and figures (privacy-attack sections skimmed). Used: super-linear relation
between duplicate count and regeneration rate (10 copies → ~1,000× more generations).

**Carlini, Ippolito, Jagielski, Lee, Tramèr, Zhang 2023. "Quantifying Memorization Across Neural Language
Models." arXiv:2202.07646.** Read: §1–3. Used: three log-linear memorization factors (scale 2–5×,
duplication, context); GPT-J memorizes ≥ 1 % of the Pile.

**Hernandez et al. 2022. "Scaling Laws and Interpretability of Learning from Repeated Data." arXiv:2205.10487.**
Read: abstract, §1–2 summary of results. Used: double-descent harm from a small repeated subset; harm peaks when
train loss on the repeated subset approaches zero (diagnostic); copying damaged up to 3× effective size.

**Tirumala, Simig, Aghajanyan, Morcos 2023. "D4: Improving LLM Pretraining via Document De-Duplication and
Diversification." arXiv:2308.12284.** Read: abstract, §1. Used: embedding-based selection on top of MinHash
dedup gives ~20 % efficiency and up to 2 % downstream at 6.7B; repeating selected data beats repeating random data.

**Abbas, Tirumala, Simig, Ganguli, Morcos 2023. "SemDeDup." arXiv:2303.09540.** Read: abstract, §1. Used:
semantic duplicates (embedding k-means, within-cluster radius) can be removed (50 % of LAION) with small loss.
Counter-evidence: DCLM Table 4 (SemDedup on text hurt).

**Gerlach & Font-Clos 2018. "A standardized Project Gutenberg corpus." arXiv:1812.08092.** Read: full paper.
Used: PG has the same book under different IDs (different editions); SPGC removes only duplicate files of the
same ID, not content duplicates; header/footer removal by a list of known markers
(c-w/gutenberg `strip_headers.py`); 81 % of PG books tagged English, 56 languages, ≈0.3 % multi-language tags.

**Bandy & Vincent 2021. "Addressing 'Documentation Debt' in ML Research: A Retrospective Datasheet for
BookCorpus." arXiv:2105.05241.** Read: §1–4. Used: book-corpus duplicate structure (identical, truncated,
slightly edited copies); 98 empty files; 655 files truncated below the stated 20k-word minimum; counts of
repeated copyright sentences (n=788, 111, 109, ...).

Cross-references: Gopher (A.1.1 dedup), RefinedWeb (G.3), FineWeb (§3.4, E), DCLM (App. L), Dolma (§5.4),
CCNet, ROOTS, IB 1.0 (§4.6), IB-ET (§4.3, §4.10), WIMBD (§4.3) — entries below.

---

## (b) Extraction noise from PDFs and OCR

### Patterns
- PDF stores glyphs with positions, no reading order (olmOCR Fig. 2). Multi-column pages, floats,
  footnotes break order.
- Running headers, footers, page numbers inside the text at every page break (olmOCR HF tests; IB 1.0 §4.9;
  IB-ET §4.6–4.7). Page numbers also appear as lone numeric lines; tables produce many numeric lines
  (IB-ET Fig. 7: 8.7 "page numbers" per page in a ready-reckoner).
- End-of-line hyphenation: `informa- / tion`; English books average 4.9 EOL hyphens merged per page;
  89 % merge, 7 % keep hyphen, 4 % hyphen + space (IB-ET Table 4).
- Tables and graphs OCR into fragments; low "tokenizability" (< 30 for English) flags table/graph/music
  volumes (IB 1.0 §4.8). Orphan table lines ("Do. Do. Do.", "Value. Quantity.") form huge duplicate clusters (IB-ET).
- Math: Tesseract-style OCR flattens sub/superscripts; GROBID stores formulas as Unicode and misses inline
  math (Nougat §1, §5). Text with math removed leaves dangling sentences (OpenWebMath Fig. 4 example:
  "our answer for that day will be. This problem...").
- Neural OCR (Nougat, olmOCR, Marker's models) degenerates into repeated sentences/lines: 1.5 % of in-domain
  pages, more out-of-domain (Nougat §5.4); olmOCR's most common failure (App. D.2). olmOCR-Bench baseline test
  flags a repeated n-gram run longer than 30 at the end, and CJK/emoji characters as "language switching".
- Hallucinated or captioned content; GPT-4o omits or rewrites dense pages (olmOCR §2.2).
- Endmatter: title page, copyright page, TOC, index, bibliography — ~14 pages/volume (IB-ET §4.4).
- Character-level OCR errors (`irom`, `tho`, split/merged words) degrade LM representations, worse at lower
  OCR quality (Todorov & Colavizza).

### Detection / fix methods with parameters
- Unigram log-likelihood filter (Common Pile/peS2o; see §0): catches only extreme OCR garbage at −20.
- OCRoscope: cld2 language on rolling 7-grams vs. document language; share of mismatched/unknown 7-grams =
  noise (example: 41 % unrecognized → quality 59 %) (Common Corpus App. F.2). OCRerrcr: token-level error tagger.
  OCRonos (Llama-3-8B) correction; may rewrite badly degraded text; occasional repeated words (App. F.4).
- Dehyphenation by ranking 3 options (merge / keep hyphen / hyphen + space) with char 1–5-gram LMs (corpus
  base model + book-specific model), add-k (k = 0.001), Stupid Backoff 0.4, 10-char window each side (IB-ET §4.5).
- Running header/footer removal: candidates = top/bottom 5 lines of each page, length ≥ 6 chars; MinHash
  128 perms over char 5-grams, LSH Jaccard 0.85; remove if ≥ 3 near-duplicates within a 5-page span; separate
  indices for headers and footers (IB-ET §4.6). Requires page boundaries.
- Page numbers: line in top/bottom 5, ≤ 8 chars, ≥ 1 numeric char, ≤ 1 non-numeric non-space char; later
  remove "sentences" that are a single short number (IB-ET §4.7).
- Line-type classifier (PARAGRAPH_CHUNK, PAGE_NUMBER, RUNNING_HEAD, NOISE, ...; 71 % accuracy) + heuristics to
  reflow paragraphs; −0.97 % chars, English mean sentence length −46 chars, tokenizability +4.6 (IB 1.0 §4.9).
- Endmatter page classifier (F1 0.97) separating TOC/index, bibliography, other (IB-ET §4.4).
- Repetition detection for OCR output: Gopher repetition table (§d); Nougat logit-variance heuristic (window
  B = 15, threshold 6.75) needs model logits, so not applicable post hoc; olmOCR end-of-text repeated n-gram > 30.
- Re-extraction: olmOCR (document anchoring + fine-tuned Qwen2-VL-7B) scores 75.5 vs **Marker v1.7.5 70.1**
  on olmOCR-Bench; Marker passes 84.9 % of header/footer-absence tests, 27.8 % of old-scan tests, 57.6 % tables,
  76.0 % arXiv math (Table 4). Replacing Grobid+rules peS2o with olmOCR text gave +1.3 average points on OLMo-2-7B
  continued pretraining (Table 5).
- BPB per paragraph with a small LM (Qwen3-0.6B-Base) as a predictability signal; compare **within** a book
  because genre and math raise BPB (IB-ET §4.11).

### Papers
**Poznanski et al. 2025. "olmOCR: Unlocking Trillions of Tokens in PDFs with Vision Language Models."
arXiv:2502.18443.** Read: full paper and Appendices B–E. Used: benchmark categories and Marker scores above;
unit-test style checks (text absence for headers/footers within first/last N chars; reading order; tables;
math; baseline repetition/charset test); normalization before comparison (NFC, quotes/hyphens to ASCII,
whitespace); olmOCR-peS2o +1.3 points; prompt rules "remove headers and footers, keep references and footnotes,
equations in LaTeX, tables in markdown".

**Blecher, Cucurull, Scialom, Stojnic 2023. "Nougat: Neural Optical Understanding for Academic Documents."
arXiv:2308.13418.** Read: full body. Used: math loss in classical OCR/GROBID; repetition collapse rate 1.5 %
and why (after a mistake); anti-repetition augmentation; logit-variance detector; page-level processing causes
cross-page inconsistencies (bibliography, section numbers).

**Lo, Wang, Neumann, Kinney, Weld 2020. "S2ORC: The Semantic Scholar Open Research Corpus." arXiv:1911.02782.**
Read: §1–3. Used: GROBID removes headers, footers, equations, tables from body text; filters: no title, no
authors, < 100 chars of text, not English (cld2 score 0.9); containment-based title matching (§a).

**Todorov & Colavizza 2022. "An Assessment of the Impact of OCR Noise on Language Models." arXiv:2202.00470.**
Read: full paper. Used: models trained on OCR text diverge from ground-truth-trained ones, more at lower OCR
quality (German/Dutch worst); this motivates OCR-quality filtering rather than relying on scale.

**Langlais et al. 2025/2026. "Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training."
arXiv:2506.01732.** Read: §1–5, App. F–G. Used: OCRoscope / OCRerrcr / OCRonos / Segmentext; Presidio PII with
custom phone regex (55–60 % → 85 % phone recall), replacement by fictitious realistic values instead of tags;
Celadon toxicity classifier for historical text; document metrics (char repetition, TTR, top-word proportion,
alphanumeric, whitespace, uppercase fractions); Wikimedia templates/transclusion problem and use of Wikimedia
Enterprise rendered-HTML dumps.

**Cargnelutti et al. 2025. "Institutional Books 1.0: A 242B token dataset from Harvard Library's collections."
arXiv:2506.08300.** Read: §1–4.9. Used: text-level (per-chunk) LID finds side-by-side translations; SimHash
book dedup with 15 % length rule; OCRoscope vs. GRIN OCR scores (means 88.16 vs 88.38); tokenizability metric;
line-type post-processing results.

**Lowry-Duda, Cargnelutti, Brobston, Ismail, Leppert, Watson, Zittrain 2026. "Institutional Books — Enriched
Text: A customizable multilingual open-source pipeline for denoising, deduplicating, and annotating OCR text at
scale." arXiv:2608.19026** (found by a web search for running-header detection). Read: §1–4.11.
Used: soft vs. hard Unicode normalization (hard = NFKC + ASCII quotes/spaces, used only for matching);
duplicate-page SimHash; endmatter separation; dehyphenation; header/footer and page-number rules; paragraph
SimHash dedup; library-fine boilerplate cluster (21,056 copies) removed; BPB annotation; preference to
annotate rather than delete.

Cross-references: Gopher repetition rules (§d), ROOTS tokenizer-ratio outliers (§d), PG-19 (§f).

---

## (c) HTML / wikitext markup and navigation chrome

### Patterns
- WET-style text keeps menus, footers, cookie/consent text (Pile C.1.1; FineWeb §3.2; DCLM App. K: WET docs ≥ 2×
  longer than resiliparse/trafilatura output).
- Leftover lines: social counters ("3 likes"), "Read more", "sign-in", "items in cart", all-caps or numeric-only
  lines, one-word lines (RefinedWeb G.2).
- Uncleaned HTML/markdown in top n-grams: `&amp;`, `< br / >`, `[1][2][3]` reference markers, "( Log Out / Change )"
  (WIMBD Table 3). Wikipedia citation markers `[1]`, `[citation needed]` (C4).
- Wikitext: templates (convert, nihongo, lang) dropped by mwparserfromhell-based extraction → broken sentences
  with missing entities; References/External links leave short phrase lists (Wiki-40B Table 1); macros for units,
  dates, language notation; sort keys and "Edit" buttons leaking from HTML (WikiText §4.3).
- Math in HTML: MathJax delimiters in raw HTML (rendered only by JavaScript), `<math>` with
  `annotation encoding="application/x-tex"`, MathML, images with LaTeX in alt text or URL (latex.codecogs.com),
  WordPress plugins; most extractors drop these (OpenWebMath §3.4). libretexts.org is a top-20 OpenWebMath domain.

### Detection / fix methods
- Main-content extraction with trafilatura (XPath exclusion of nav, readability/jusText fallbacks) or resiliparse
  (8× faster, ~10 % more text) (trafilatura paper; DCLM §4.2: both ≥ 2.5 CORE points over WET). jusText discards
  more text (Pile C.1.2); Nemotron-CC got 57.4 % more high-quality tokens with jusText + no heuristic filters
  without accuracy loss.
- Line-wise correction rules + discard doc if flagged words > 5 % (RefinedWeb G.2).
- C4 line rules: drop lines with "javascript", "terms of use", "privacy policy", "cookie policy", "uses cookies";
  remove citation markers; drop pages with "lorem ipsum" or `{` (C4 §2.2). FineWeb kept all C4 rules except
  terminal punctuation (that one removed ~30 % of tokens).
- Wikipedia: remove non-content sections (See also, References, External links), images, captions, tables,
  lists; drop disambiguation, redirect, deleted, non-entity (Wikidata) pages; keep article/section/paragraph
  markers (Wiki-40B). DCLM removed "See Also" and "References" titles (present in 90 % of articles) from its
  Wikipedia classifier positives. Dolma: WikiExtractor, drop pages ≤ 25 words (short templated pages or parse
  errors). WikiText: only Good/Featured articles; list-heavy sections removed; formulas → `<formula>`.
- Prefer rendered HTML (Wikimedia Enterprise) to raw wikitext to keep transcluded text (Common Corpus §4.4).
- Math: detect MathJax config and delimiters; extract all 7 encodings to LaTeX; escape stray `$`; keep
  docs with MathScore > 0.17 if LaTeX found (OpenWebMath).
- Tokenizer bytes-per-token as a format-outlier signal per sub-source (ROOTS §4.3: Wiktionary = foreign-language
  definitions, Wikiversity = technical terms + LaTeX, Wikivoyage = tables as text, PG = `*******` page ends).

### Papers
**Barbaresi 2021. "Trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and
Extraction." ACL 2021 demo (aclanthology 2021.acl-demo.15).** Read: full paper. Used: cascade of XPath
exclusion/inclusion, relevance checks (element type, text length, link density), fallbacks (readability,
jusText), baseline extraction; evaluation on 500 pages; jusText = length, link and stop-word densities;
boilerpipe = shallow features with classifiers.

**Merity, Xiong, Bradbury, Socher 2016. "Pointer Sentinel Mixture Models" (WikiText). arXiv:1609.07843.**
Read: §4 (dataset); model sections skimmed. Used: WikiText construction rules; WikiText-2/103 share validation
and test articles (contamination relevance, §h).

**Guo, Dai, Vrandečić, Al-Rfou 2020. "Wiki-40B: Multilingual Language Model Dataset." LREC 2020
(aclanthology 2020.lrec-1.297).** Read: §1–3. Used: page and section filters above; TF-Datasets/mwparserfromhell
template-deletion errors; split by hash of Wikidata ID to avoid leakage.

**Paster, Dos Santos, Azerbayev, Ba 2023. "OpenWebMath." arXiv:2310.06786.** Read: §1–4. Used: math encodings,
MathJax extraction, line processing, KenLM perplexity cut (ProofPile model, > 15,000 removed), SimHash dedup
0.7, manual inspection of top domains and longest documents.

Cross-references: C4 (T5 §2.2), RefinedWeb (G.2), DCLM (App. K), Pile (C.1), WIMBD, Common Corpus.

---

## (d) Low-quality or non-prose content (tables, lists, references, indexes, stubs, repetition)

### Patterns
- Keyword lists / SEO, machine-generated pages, symbol-heavy text, bullet-only pages, ellipsis endings (Gopher).
- Short lines dominate low-quality docs; lines without terminal punctuation (FineWeb Fig. 8: low-quality docs
  concentrate at fraction-of-lines-ending-with-punctuation < 0.12).
- Character repetition `-----`, `*****`, `??????????` among top 13-grams/10-grams (Pile E.1; WIMBD).
- Reference lists as `[1][2][3]` sequences (WIMBD, S2ORC/peS2o).
- Wikipedia stubs and template pages ≤ 25 words (Dolma), list pages and infobox-heavy non-entity pages (Wiki-40B).
- Heuristic web filters remove legitimate high-quality text: the C4/Gopher/KenLM pipeline removed 18.1 % of
  tokens that the FineWeb-Edu classifier rated high quality (Nemotron-CC Table 1).

### Detection / fix methods with thresholds
- Gopher quality rules: 50–100,000 words; mean word length 3–10 chars; `#` or `...` to word ratio ≤ 0.1;
  ≤ 90 % lines starting with a bullet; ≤ 30 % lines ending with an ellipsis; ≥ 80 % words with an alphabetic
  char; ≥ 2 of {the, be, to, of, and, that, have, with}.
- Gopher repetition thresholds (Table A1): duplicate line fraction 0.30; duplicate paragraph fraction 0.30;
  duplicate line char fraction 0.20; duplicate paragraph char fraction 0.20; top 2/3/4-gram char fraction
  0.20/0.18/0.16; duplicate 5…10-gram char fraction 0.15/0.14/0.13/0.12/0.11/0.10.
- FineWeb custom rules: lines-ending-with-punctuation ratio ≤ 0.12 removes (10.14 % tokens); duplicated-line
  char ratio ≥ 0.1 removes (12.47 %); fraction of lines < 30 chars ≥ 0.67 removes (3.73 %); together ~22 % tokens,
  +~1 % aggregate. Thresholds chosen by comparing histograms of a high- vs low-quality set (method, §3.6).
- C4: keep lines ending in terminal punctuation; ≥ 5 words per line; ≥ 3 sentences per page.
- Dolma: C4 NoPunc (drop docs with > half lines not ending in `. ? ! "`) + Gopher All was best; token sequence
  repeated > 100 times → drop (used for PG and peS2o too).
- MADLAD-400 "questionable sentence" score: sentence-level LID ≠ doc LID; list case (> 50 % capitalized tokens,
  ≥ 12 tokens); length < 20 or > 500 chars; > 20 % chars in `[0-9{}+/()>]`; "cursed" regex hits; drop doc if
  > 20 % sentences questionable or < 5 sentences.
- ROOTS: character/word repetition, special-character ratio, closed-class-word ratio, flagged words, KenLM
  perplexity, min words; thresholds set per language by native speakers with a visualization tool.
- Model-based: fastText bigram classifier (OH-2.5 + ELI5 positives, top 10 %) best in DCLM; FineWeb-Edu
  classifier (Llama-3-70B annotations, threshold 3, F1 82 %) +MMLU 33→37 %, ARC 46→57 %; CCNet KenLM terciles;
  peS2o/Common Pile unigram filter. Nemotron-CC: do not apply heuristic filters to classifier-high-quality text
  (+2 MMLU in ablation). Muennighoff: perplexity filtering helped downstream; filtering helps most on noisy data.
- Human judgments are a poor proxy: AskLLM had 82 % ROC-AUC with human labels but worse CORE (DCLM App. N).

### Papers
**Rae et al. 2021. "Scaling Language Models: Methods, Analysis & Insights from Training Gopher."
arXiv:2112.11446.** Read: §1–3, §4.1–4.3, §5.1 start, Appendix A (all), D.2–D.6. Used: all MassiveText rules
and thresholds above; dedup (13-gram Jaccard 0.8); test-set filtering (§h); SafeSearch instead of word lists;
NFKC normalization is lossy (exponents) and should be avoided (A.4); ablation that each stage helps (A.3.2);
document sampling crops of 15 × n bytes (A.1.2); Books sampled at 27 %.

**Raffel et al. 2020. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (C4).
arXiv:1910.10683.** Read: §2.2, §3.4, §3.5.2. Used: C4 rules; langdetect ≥ 0.99; unfiltered C4 worst on every
task (Table 8); domain-matched data helps matching tasks (Wikipedia + TBC → MultiRC EM 25.78 → 50.93); repeating
a dataset 64× costs little, 256×+ degrades (Table 9); temperature-scaled mixing (§i).

**Penedo et al. 2024. "The FineWeb Datasets." arXiv:2406.17557.** Read: §1–5, App. D–F. Used: extraction (trafilatura
on WARC > WET), base filtering (fastText English ≥ 0.65, Gopher rules), per-snapshot MinHash, C4 rule ablation,
custom rules, FineWeb-Edu prompt and classifier, Table 2 list of other filters tested.

**Li et al. 2024. "DataComp-LM." arXiv:2406.11794.** Read: §1–6, App. J, K, L, M, N, O. Used: resiliparse vs
trafilatura vs WET; BFF dedup algorithm and parameters; fastText classifier recipe (400k examples, bigrams,
top 10 %); mixing high-quality sources into DCLM-Baseline **hurt** (Table 6: −1.2 CORE) even when those sources
were classifier-filtered (Table 24); decontamination methods (§h).

**Penedo et al. 2023. "The RefinedWeb Dataset for Falcon LLM." arXiv:2306.01116.** Read: §1–3, App. G, H.
Used: line-wise filter rules; MinHash and EXACTSUBSTR variants; URL dedup across dumps; largest clusters are
templated licenses and sitemap notices.

**Wenzek et al. 2020. "CCNet." arXiv:1911.00359.** Read: full paper. Used: paragraph-hash dedup normalization,
dedup-then-LID ordering, fastText LID > 0.5, 5-gram KenLM perplexity terciles per language; perplexity removes
keyword lists but also penalizes valid text far from Wikipedia (forums, jargon).

**Su et al. 2024. "Nemotron-CC." arXiv:2412.02595.** Read: §1–3 (ablations), skimmed appendices. Used: heuristic
filters remove 18.1 % of high-quality tokens; classifier ensemble with 20 score buckets; rephrasing (not
applicable here: Common Pile excluded LLM-generated text, §2.1 of 2506.05209).

**Kudugunta et al. 2023. "MADLAD-400." arXiv:2309.04662.** Read: §1–2 (data, filters, self-audit); evaluation
skipped. Used: pct_questionable filter and thresholds; audit protocol (20 docs per language); issues found
(boilerplate, misrendered text, single-source domination).

**Laurençon/Piktus et al. 2023. "The BigScience ROOTS Corpus." arXiv:2303.03915.** Read: §1–4. Used: per-domain
menu removal (lines in > 1 % of pages); filter families; SimHash 6-grams Hamming ≤ 4 with long-doc exception;
suffix-array substring dedup found 21.67 % duplicated bytes; regex PII (§g); tokenizer-ratio analysis.

---

## (e) Language mixing

### Patterns
- Non-English passages inside English documents: bilingual/side-by-side editions (IB 1.0 Fig. 7: a "Latin" book
  is 61 % French, 38 % Latin); Wiktionary definitions of foreign words (ROOTS); English boilerplate inside
  non-English pages and vice versa (CCNet).
- LID failure modes on noisy text: badly encoded PDFs, misrendered non-Unicode text, invalid HTML, URLs, soft
  hyphens, look-alike Unicode characters all get assigned to some language (Caswell et al. §3).
- Under class imbalance, high balanced-set precision gives low in-the-wild precision:
  precision_crawl = x·r / (x·r + (1 − x)·f) (Caswell et al.).
- Dolma audit: after English filtering 0.86 % of docs are "un" (unidentified), 0.06 % Chinese.

### Detection / fix methods
- Document-level fastText English score > 0.5 (Common Pile, CCNet, Dolma); Dolma checked that 0.5 keeps all
  International Corpus of English documents from nine countries (App. G); FineWeb uses ≥ 0.65.
- Dolma for PG: LID per paragraph, average the scores, drop book if average < 0.5.
- Document consistency: LID per sentence, majority = document language, keep only matching sentences
  (Caswell et al. §3); MADLAD uses the mismatch as a "questionable" signal.
- Percent-threshold wordlist filtering (keep sentences with ≥ x % in-vocabulary words; x = 20 % in Table 4) and
  disjunctive TF-IDF-IIF lists (Caswell et al.).
- Text-level LID on 768-char chunks with trigram conversion (robust to hyphenation/line breaks) (IB 1.0 §4.4).
- Per-paragraph language annotation (IB-ET §4.12).
- Self-audit of samples with a label taxonomy CC/CB/CS/X/WL/NL (Kreutzer et al.).

### Papers
**Caswell, Breiner, van Esch, Bapna 2020. "Language ID in the Wild." arXiv:2010.14571.** Read: §1–6. Used:
failure-mode list; precision formula; wordlist filters; document-consistency filtering.

**Kreutzer et al. 2022. "Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets." arXiv:2103.12028.**
Read: §1–5. Used: audit method (100 sentences per corpus), taxonomy (correct, boilerplate, short, wrong language,
non-linguistic), non-expert annotations agree with experts at 0.98 (6-class) on OSCAR and 0.79 (binary) on CCAligned.

Cross-references: Dolma (App. G, K, N.4), CCNet, MADLAD-400, IB 1.0 (§4.4), Common Corpus (OCRoscope).

---

## (f) License and attribution boilerplate

### Patterns
- Per-chapter/per-page license statements, attribution blocks, "published under CC BY-SA" lines (Common Pile
  PDR line list; DOAB front/back-matter license statements used as an inclusion test).
- Copyright notices repeated across books (BookCorpus counts, §a).
- PG license header/footer and "produced by ... Distributed Proofreaders" credits (PG-19; SPGC; Common Pile
  repo docstring says producer preambles were kept).
- Licenses are the largest MinHash clusters on the web (RefinedWeb H.1); licenses are "repeated for structural
  (e.g., legal, economical) reasons" (Lee et al. Ethics).
- Wrong or laundered license metadata: 70 %+ of dataset licenses "unspecified", 66 % of analyzed HF licenses in
  a different (often more permissive) category than the author's (Data Provenance).
- Removing duplicated attribution strings can make a model learn content without attribution (Lee et al. §7).

### Detection / fix methods
- Exact-line lists per source (Common Pile [repo]); fragile to any variation (whitespace-stripped exact match).
- Paragraph-hash dedup with normalization (CCNet) and per-collection frequency (> 1 % of pages, ROOTS) catch
  variants with different titles/years/numbers.
- EXACTSUBSTR ≥ 50 tokens removes repeated license spans across documents (RefinedWeb).
- Keep license/attribution in metadata (all Common Pile sources carry `metadata.license`), not in the token stream.
- PG markers: SPGC marker list; Common Pile regex; PG-19 "removing boilerplate license text".

### Papers
**Rae, Potapenko, Jayakumar, Lillicrap 2020. "Compressive Transformers for Long-Range Sequence Modelling" (PG-19).
arXiv:1911.05507.** Read: §4 (PG-19 benchmark) and App. C–D; model sections skimmed. Used: PG-19 = PG books
published before 1919, 28,752 books (train 28,602, valid 50, test 100); minimal preprocessing removed license
boilerplate and **replaced discriminatory words with a unique `<DWx>` token** (Ofcom list). The Common Pile adds
PG-19 books, so its PG text may come from the PG-19 version (check for `DW` tokens) and it contains the PG-19
validation/test books (§h).

**Longpre et al. 2023. "The Data Provenance Initiative." arXiv:2310.16787.** Read: abstract, §1. Used: license
omission 70 %+, error rates 50 %+, 66 % miscategorized, typically more permissive.

Cross-references: Common Pile §2.1/App. C.1, Gerlach & Font-Clos, BookCorpus datasheet, RefinedWeb, Lee et al.

---

## (g) PII

### Patterns
- E-mails, phones, IPs; estimated 200M e-mails, 4B phone numbers, 97M IPs in mC4-en (WIMBD); e-mails abundant in
  S2ORC/peS2o (Common Corpus citing WIMBD).
- Duplicated PII leaks more; scrubbing reduces but does not prevent leakage; NER scrubbing recall 97 % for names,
  80 % for care-unit numbers in clinical text; scrubbing raises perplexity similar to DP (Lukas et al.).
- Regex false positives: e-mail regex F1 81.8 % → 96.83 % after fixes; key/secret regex precision < 4 %;
  invalid or private IPs and DNS-server IPs are not PII (StarCoder §4).
- Placeholder tokens change text format (`|||EMAIL_ADDRESS|||` in Dolma, `<EMAIL_ADDRESS>` in Common Pile);
  Common Corpus argues tags "could undermine the model's understanding" and uses fictitious values.

### Methods
- Dolma: regex for e-mail, IP, phone; ≤ 5 spans → replace with tokens; ≥ 6 spans → drop document; 0.02 % /
  0.001 % of CC docs. Execution choice had no effect on model performance. (Note: in the extracted datasheet text the
  regexes printed under "IP addresses" and "Phone numbers" appear swapped.)
- Common Pile: regex replacement only (no document drop) [repo `pii_regex_with_counts_fast_v2`, min_score 0.1].
- ROOTS: regex KEY/EMAIL/USER/IP_ADDRESS, skipping years and simple numbers.
- StarCoder: NER model (StarEncoder) + pseudo-labels; F1 > 90 % names/emails/IPs; post-filters (secrets ≥ 4
  chars, keys ≥ 9 chars and gibberish, valid public IPs only, names need a space); placeholders `<NAME>`, `<EMAIL>`,
  `<KEY>`, `<PASSWORD>`; random private IPs for IPs.
- Presidio + custom phone regex (Common Corpus).
- olmOCR LLM prompt: decide whether the document is meant for public dissemination; names/e-mails/phones are
  identifiers; addresses/biographical data count only with an identifier; government IDs, financial data always.

### Papers
**Lukas et al. 2023. "Analyzing Leakage of Personally Identifiable Information in Language Models."
arXiv:2302.00539.** Read: §1–2. Used: scrubbing limits, utility cost, duplication increases leakage.

**Li et al. 2023. "StarCoder: may the source be with you!" arXiv:2305.06161.** Read: §4 (PII redaction) only.
Used: regex vs NER numbers, post-processing rules, placeholder scheme.

**Elazar et al. 2024. "What's In My Big Data?" arXiv:2310.20707.** Read: §1–4 main text. Used: PII counts; also
duplicates (MD5 exact: Pile > 60 %, RedPajama ~50 %), empty documents in 4 of 10 corpora, length-histogram outliers
(e.g. Pile docs of exactly 8,194 chars from truncated DM Mathematics), top n-grams, contamination search (§h).

Cross-references: Dolma (§5.3, App. I), ROOTS (§3.3), Common Corpus (§5), olmOCR (E.4), Common Pile (§4.1).

---

## (h) Benchmark contamination

### Patterns
- Benchmarks built from Wikipedia/web are present as inputs: QNLI inputs 2 %–> 50 % in C4; generation targets
  (XSum, TIFU, WikiBio, AMR) 1.87–24.88 % in C4 (Dodge et al.).
- Whole datasets in code/web repos: WSC, COPA, AX, SICK, SemEval 100 % in Dolma (mostly GitHub); HumanEval,
  WiC, e-SNLI, SNLI > 90 % (Dolma App. L).
- 4.6 % of C4 validation and 14.4 % of RealNews validation have near-duplicates in train (Lee et al.).
- Wikitext-103 pages appear in Wikipedia dumps; edits after collection defeat n-gram matching (Gopher A.1.1).
- 1 %–45.8 % of examples of common benchmarks appear verbatim in Common Crawl; contamination grows over time;
  effect on scores varies (C-Eval +14, HellaSwag +7, MMLU ~0) (Li et al. 2023).

### Methods with thresholds
- GPT-3: any 13-gram overlap; PaLM: ≥ 70 % of 8-grams found; GPT-4: ≥ 50-char substring; Llama 2: tokens in
  ≥ 10-gram overlaps, "dirty" if > 80 % of tokens contaminated (all as summarized in Li et al. 2023 and DCLM App. O).
- Gopher: 13-gram Jaccard > 0.8 between train and test docs, plus removal of the exact Wikitext-103
  validation/test pages from the Wikipedia training set.
- Dodge et al.: exact match after normalizing case and punctuation.
- Dolma: paragraph > 13 Unicode tokens that appears in Paloma → drop the document (≤ 0.02 % of docs).
- DCLM: flag pages containing the question and ≥ 1 answer option (MMLU: last sentence of the question);
  removing them did not lower scores.
- Li et al. 2023: METEOR recall > 0.75 between query and matched page (order penalty 0.8).
- Oren et al.: model-side test — canonical benchmark order more likely than shuffled order → contamination, with
  false-positive control; detects test sets seen twice in pretraining for a 1.4B model.
- Pile/GPT-3: 13-gram decontamination before evaluation comparisons.

### Papers
**Dodge et al. 2021. "Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus."
arXiv:2104.08758.** Read: §1–5. Used: contamination numbers; exact-match normalization; blocklist filtering
disproportionately removes text from and about minority individuals (argues against word blocklists, also cited
by Gopher); machine-translated + OCRed patents as hidden low-quality source.

**Li, Guerin, Lin 2023. "An Open-Source Data Contamination Report for Large Language Models." arXiv:2310.17589.**
Read: §1–4. Used: threshold survey and METEOR method; input-only vs input-and-label contamination.

**Oren, Meister, Chatterji, Ladhak, Hashimoto 2023. "Proving Test Set Contamination in Black Box Language
Models." arXiv:2310.17623.** Read: §1–2. Used: exchangeability test as a post-training audit.

Cross-references: Gopher (A.1.1, D.4), Lee et al., Dolma (App. L), DCLM (§4.6, App. O), WIMBD (§4.4), Pile (§4.1),
WikiText (shared valid/test articles), PG-19 (valid/test books).

---

## (i) Data mixing and upsampling of small high-quality sources

### Patterns / findings
- Up to 4 epochs ≈ unique data; value of repeats decays with half-life R*_D ≈ 15 repetitions (16 epochs); beyond
  that returns vanish (Muennighoff et al.). 8.7B model with 4 epochs: +0.5 % validation loss vs 1 epoch.
- Repeated small subsets can cause disproportionate harm (Hernandez et al.); heavy repetition of very high-quality
  sources (Wikipedia, books) is flagged there as a risk for large models.
- Pile capped epochs at 3 and mostly ≤ 2 (App. C.22); Gopher fixed Wikipedia at 2 % so it gets one epoch; books
  at 27 % helped LAMBADA (A.3.1).
- DoReMi: proxy-model Group-DRO weights (280M → 8B) +6.5 points few-shot, 2.6× faster; Wikipedia was
  downweighted yet Wikipedia-based tasks improved; low- and high-entropy domains can be downweighted.
- DCLM: adding Wikipedia/books/arXiv/GitHub to a strongly filtered web set hurt (−1.2 CORE).
- Dolma: small fractions of in-domain data suffice for domain fit (4.9 % vs 24.2 % papers gave same S2ORC ppl).
- Temperature-scaled mixing (T = 2 best) and size cap K (T5 §3.5.2).
- Common Pile used 6 repeats for small sources in 1T and 12 (up to 16 for some) in 2T; their own text says this
  "may result in diminishing returns".
- Duplicated boilerplate × upsampling multiplies repeat counts: a license line in every PressBooks chapter, at 6
  repeats, is seen (#chapters × 6) times (inference from Hernandez/Kandpal patterns + Common Pile Table 7; to be
  measured on the data).

### Papers
**Muennighoff et al. 2023. "Scaling Data-Constrained Language Models." arXiv:2305.16264.** Read: §1–7, App. E.
Used: repeat value formula D' = U_D + U_D R*_D (1 − e^{−R_D/R*_D}); R*_D ≈ 15; 4-epoch rule; code up to 50 % no
harm on NL tasks; perplexity filter helps, dedup filter did not (downstream).

**Xie et al. 2023. "DoReMi." arXiv:2305.10429.** Read: §1–4. Used: algorithm (excess-loss Group DRO, η = 1,
smoothing 1e-3, average weights over training), results above.

**Gao et al. 2020. "The Pile." arXiv:2101.00027.** Read: §1–5, App. C, D, E. Used: epochs per component; PG-19 and
Wikipedia used "with minimal preprocessing"; Wikipedia = TFDS 20200301.en with title prepended; jusText choice;
MinHashLSH 10 hashes Jaccard 0.5 within OWT2 and Pile-CC only (no Pile-wide dedup); 13-gram statistics of CC
(styling characters and HTML entities dominate top 13-grams).

**Soldaini et al. 2024. "Dolma." arXiv:2402.00159.** Read: §1–9, App. G–N (datasheet preprocessing). Used: web
pipeline order (URL dedup → doc dedup → quality/content filters → paragraph dedup); Gopher + C4 NoPunc ablation;
toxicity thresholds (0.4 vs 0.0004: low threshold removes 29.1–34.9 % and performs better, they kept 0.4);
toxicity scores are bimodal near 0 and 1; PII rules; PG (per-paragraph LID average, ≥ 25 words, repeated-token
filter, dedup by exact title); Wikipedia (WikiExtractor, ≤ 25 words removed); decontamination; mixing results;
filters are weakly correlated, so they are complementary (App. J).

Cross-references: Common Pile §4.2/App. K–L, Gopher A.3.1, T5 §3.4.2/§3.5.2, DCLM §4.5/App. M, Hernandez et al.

---

## Source-specific patterns to check in the raw downloads (derived from the papers above; not yet verified on data)

| Source | Patterns to measure | Literature method |
|---|---|---|
| doab (Marker PDF→markdown) | running headers/footers, page numbers, EOL hyphens, repeated lines from OCR loops, markdown tables, image placeholders, LaTeX fragments, TOC/index/bibliography, front-matter license pages, non-English chapters, exact/near/partial duplicate books (editions, chapter-level PDFs) | IB-ET §4.4–4.7, §4.10; Gopher repetition table; olmOCR baseline repetition test; OCRoscope; containment metric; per-paragraph LID |
| libretexts (HTML) | math dropped or as MathJax source, dangling sentences, attribution/license blocks per section, navigation and "Learning Objectives"/exercise templates, remixed duplicate textbooks | OpenWebMath extraction; ROOTS >1 % repeated-line removal; CCNet paragraph hashing; MinHash + containment; contamination check vs every evaluation set in use (e.g. MMLU, ARC) |
| pressbooks (HTML) | per-chapter license/attribution boilerplate, front matter, short pages (51 % of docs survived all upstream filters; length threshold 600 chars), adapted copies of the same book | same as libretexts; Hernandez/Kandpal repeat-count risk under 6× upsampling |
| oercommons (HTML) | lesson-plan templates, metadata/navigation text, short pages (56 % of docs survived all upstream filters; length threshold 300 chars) | RefinedWeb line rules; MADLAD pct_questionable; WIMBD length-histogram outliers |
| project_gutenberg | residual PG header/footer, producer credits, transcriber notes, `[Illustration]`, `*  *  *`, page-number artefacts, `<DWx>` tokens from PG-19, multiple editions/translations of the same work, bilingual editions, PG-19 valid/test books | SPGC marker list; Dolma per-paragraph LID; IB 1.0 SimHash 15 % rule; PG-19 split for decontamination |
| public_domain_review | CC BY-SA footer line, image captions/credits, very small source (1,412 docs) repeated 6× | exact-line removal; repeat-count control |
| wikimedia | talk pages and bot/maintenance text (1,554 exact lines removed upstream — exact match only), Wiktionary/Wikivoyage/Wikisource format outliers, broken template sentences, references and list sections, stubs, foreign-language text in Wiktionary, WikiText-2/103 valid/test articles, Wikipedia-sourced benchmark inputs (QNLI, SQuAD-type) | Wiki-40B page/section filters; Dolma ≤ 25-word filter; ROOTS tokenizer ratio per wiki; Gopher removal of WikiText pages; Dodge exact-match contamination |

Note for this project: CinnamonLM's default corpus is WikiText-2 (git log "default corpus WikiText-2"). WikiText-2 and
WikiText-103 share validation and test articles (Merity et al. §4.4), so the Common Pile Wikipedia text must be
decontaminated against those articles if WikiText perplexity is reported (Gopher A.1.1 method).

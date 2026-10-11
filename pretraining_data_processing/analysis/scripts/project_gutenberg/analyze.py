#!/usr/bin/env python3
"""Quality analysis of a sample of common-pile/project_gutenberg (raw) against project_gutenberg_filtered.

Usage: python3 -I analyze.py [index|stats|dedup|filtdiff|report|all]
Sample: raw shards 00000, 00014, 00029; filtered shards 0000 (covers raw 00000 id range) and 0014 (covers raw 00029).
"""
import collections
import gzip
import json
import multiprocessing
import os
import re
import sys
import hashlib
import ast

import numpy as np
import xxhash

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
WORK = os.path.join(BASE, 'work')
RAW_SHARDS = {s: os.path.join(DATA, 'raw', 'v0', 'documents', f'{s}_pg.jsonl.gz') for s in ('00000', '00014', '00029')}
FILT_SHARDS = {s: os.path.join(DATA, 'filtered', f'project_gutenberg-dolma-{s}.json.gz') for s in ('0000', '0014')}
# Each filtered shard is compared only with the raw shard whose id range it fully covers (both are sorted by id string).
FILT_FOR_RAW = {'00000': '0000', '00029': '0014'}
N_EXAMPLES = 3


def iter_docs(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            doc = json.loads(line)
            meta = doc.get('metadata') or {}
            if isinstance(meta, str):
                meta = ast.literal_eval(meta)
            doc['metadata'] = meta
            yield doc


# ---------------------------------------------------------------- issue detectors
CREDIT_RE = re.compile(r'(?im)^[ \t]*(?:Produced by|E-?text prepared by|Etext prepared by|This e-?(?:Book|text) (?:was|is|has been) '
                       r'(?:produced|prepared|created|made|transcribed)|Transcribed (?:from .{0,80} )?by|Digiti[sz]ation by|'
                       r'Scanned (?:and proofed )?by|Prepared by|E-book prepared by)\b')
TRAILER_RE = re.compile(r'(?im)^[ \t]*(?:\*+[ \t]*)?(?:The )?(?:End of (?:the |this )?(?:Project Gutenberg|P\.? ?G\.?\b|e-?text\b)'
                        r'|Ende dieses Projekt Gutenberg|Fin (?:du|de ce|de la) (?:Projet|Project) Gutenberg'
                        r'|Fine del(?:l[ao])? Project Gutenberg|Einde van (?:dit|het) Project Gutenberg|End of Project Gutenberg)')
PG_LINE_RE = re.compile(r'(?i)project gutenberg|gutenberg\.org|ibiblio\.org|pgdp\.net|distributed proofread|gutenberg-tm|\bPGLAF\b|promo\.net/pg')
ETEXT_LINE_RE = re.compile(r'(?i)\be-?texts?\b|\bebooks?\b')
LEGAL_RE = re.compile(r'(?i)legal small print|small print!|This header should be the first thing seen|Project Gutenberg-tm|'
                      r'Gutenberg Literary Archive Foundation|START\*\*THE SMALL PRINT|We are releasing two versions of this Etext')
HTML_NOTE_RE = re.compile(r'(?m)^[ \t]*Note: Project Gutenberg also has an HTML version.*(?:\n[ \t]+\S.*)*'
                          r'|^[ \t]*Images of the original pages are available.*(?:\n[ \t]+\S.*)*')
TN_BRACKET_RE = re.compile(r"(?is)\[(?:Transcriber|Transcriber['’]s|Transcription note|End transcriber)[^\]]{0,6000}\]")
TN_PARA_RE = re.compile(r"(?im)^[ \t]*(?:Transcriber['’]?s?['’]? ?(?:notes?|remarks?)|TRANSCRIBER['’]?S? NOTES?)\b")
TN_ANY_RE = re.compile(r"(?i)transcriber['’]?s?['’]? ?notes?")
ILLUS_RE = re.compile(r'\[(?:Illustration|Frontispiece|Image|Plate|Portrait)[^\]]{0,3000}\]')
FOOTNOTE_BLOCK_RE = re.compile(r'\[Footnote[^\]]{0,6000}\]')
FN_MARK_RE = re.compile(r'\[(?:\d{1,3}|[A-Z])\]')
PAGE_RE = re.compile(r'\{\d{1,4}\}|\[(?:Pg|pg|Page|p\.)\s?[\dixvlc]+\]|<pb\b[^>]*>')
HTML_TAG_RE = re.compile(r'</?(?:i|b|em|strong|sup|sub|sc|br|p|span|div|u|tt|small|big|font|a)\b[^>\n]{0,200}>', re.I)
MOJIBAKE_RE = re.compile('Ã[\u0080-¿]|â€[\u0080-¿‘-„™œ˜]|Â[ -¿]|�')
ASTERISM_RE = re.compile(r'(?m)^[ \t]*(?:\*[ \t]*){3,}$')
CONTENTS_HEAD_RE = re.compile(r'(?m)^[ \t]*(?:TABLE OF CONTENTS|CONTENTS|Contents|INDEX|Index|LIST OF ILLUSTRATIONS|'
                              r'List of Illustrations|ILLUSTRATIONS|LIST OF PLATES|INDEX OF FIRST LINES)\.?:?[ \t]*$')
INDEX_LINE_RE = re.compile(r'(?:,|\.{3,}|\s{3,})\s*(?:\d{1,4}|[ivxlc]{1,7})(?:\s*[-–,]\s*\d{1,4})*\.?\s*$')
STUB_MEDIA_RE = re.compile(r'(?i)\.(?:mus|mp3|midi?|avi|mpg|mpeg|ly|wav|ogg|pdf|tex)\b|musical score|no text version|audio book|sheet music')
DIACRITIC_NOTE_RE = re.compile(r"\[[=)~^'`:,.]{1,2}[A-Za-z]{1,2}\]|\[(?:Greek|Hebrew|Arabic|Sanskrit|Syriac|Coptic|Gothic):[^\]]{0,500}\]")
ITALIC_RE = re.compile(r'(?<![\w_])_[^_\n]{1,200}_(?![\w_])')
BOLD_RE = re.compile(r'(?<![\w=])=[^=\n]{1,100}=(?![\w=])')
SPEAKER_RE = re.compile(r"(?m)^[ \t]*[A-Z][A-Z'\-]{1,20}(?: [A-Z][A-Z'\-]{1,20}){0,2}\.[ \t]*$")
STAGE_RE = re.compile(r'\[_?(?:Exit|Exeunt|Enter|Aside|Re-enter|Curtain|Exeunt omnes)\b')
VERSE_LINENO_RE = re.compile(r'(?m)\S[ \t]{3,}\d{1,4}[ \t]*$')
AD_RE = re.compile(r'(?i)\b(?:price|cloth|post free|crown 8vo|8vo|12mo|16mo|net|gilt|boards)\b.{0,80}(?:\d|\$|£)|\b\d+s\. ?\d*d\.')
AD_HEAD_RE = re.compile(r'(?m)^[ \t]*(?:BY THE SAME AUTHOR|BOOKS BY|ALSO BY|NEW BOOKS|NEW NOVELS|CATALOGUE OF|ADVERTISEMENTS?|PUBLISHERS[\'’] ANNOUNCEMENT)')
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
BLANKRUN_RE = re.compile(r'\n[ \t]*\n(?:[ \t]*\n)+')
HEADER_FRAG_RE = re.compile(r'\A[^\n]{0,120}\*\*\*[ \t]*\n')

ARCHAIC = set('thee thou thy thine hath doth dost hast shalt wilt ye thyself wherefore whence hither thither whither betwixt '
              'twas tis ere oft nay hither saith spake'.split())
ETH_EXCLUDE = set('teeth beneath seth elizabeth macbeth nazareth japheth kenneth gareth sheth teeth'.split())
ELISION_RE = re.compile(r"\b[a-z]{2,}'d\b")

STOP = {
    'en': 'the of and to a in that is was he for it with as his on be at by i had not are but from or have an they which you were her she there would their we him been has when who will more no if out so said what up its about into than them can only other some could these may then do any my now such like our over me even most made after also did many before must through your'.split(),
    'fr': 'le la les de des du et est un une que qui dans pour pas au aux ce cette il elle ne se sur sont avec par plus son sa ses nous vous je mais ou lui leur été être avoir comme tout'.split(),
    'de': 'der die das und ist ein eine nicht zu den dem des mit sich auf für von im sie er es ich auch als wie dass daß wird war so aber bei nach aus noch nur wenn um'.split(),
    'es': 'el la los las de del y que en un una es por con no se su para al lo como más pero sus le ya o fue este ha porque esta entre cuando muy sin sobre también'.split(),
    'it': 'il la le di che e è un una per non con del della dei delle si da in al alla gli lo ma come più anche sono suo sua era questo quella'.split(),
    'nl': 'de het een en van in is dat op te zijn met voor niet aan er die maar om ook als bij of uit nog door wat over zo hij zij ze wij'.split(),
    'pt': 'o a os as de do da dos das e que em um uma para com não por se na no mais como mas foi ao ele ela seu sua ou quando muito já'.split(),
    'la': 'et in est non ad cum ut qui quae quod sed de per enim esse sunt ab ex nec si autem hoc atque etiam inter quam'.split(),
    'fi': 'ja on ei se että hän oli ovat mutta kun niin myös tai kuin joka ole sen hänen minä sinä'.split(),
    'scand': 'och att det som en på är av för med till den har de inte om ett han men og er til ikke af jeg der'.split(),
}
STOP = {lang: set(words) for lang, words in STOP.items()}


def latin_letter(ch):
    return ch.isascii() or ('À' <= ch <= 'ɏ')


def classify_language(text):
    words = [w.lower() for w in WORD_RE.findall(text)]
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 'empty', 0.0, 0.0
    non_latin = sum(1 for ch in letters if not latin_letter(ch)) / len(letters)
    if non_latin > 0.3:
        return 'nonlatin', 0.0, non_latin
    if not words:
        return 'unknown', 0.0, non_latin
    ratios = {lang: sum(1 for w in words if w in stop) / len(words) for lang, stop in STOP.items()}
    best = max(ratios, key=ratios.get)
    if ratios[best] < 0.15:
        return 'unknown', ratios['en'], non_latin
    return best, ratios['en'], non_latin


def paragraphs_with_lines(text):
    for para in re.split(r'\n[ \t]*\n', text):
        lines = [ln.rstrip() for ln in para.split('\n') if ln.strip()]
        if lines:
            yield lines


def block_after_heading(lines, start):
    """(chars, end line) of a TOC/index block after a heading line.

    The block ends at two consecutive long prose lines, at a run of >=3 blank lines after content,
    or after 400 lines if fewer than half of its lines carry page references (guards verse texts).
    """
    total = 0
    idx = start + 1
    blank_run = 0
    content_lines = 0
    index_like = 0
    while idx < len(lines) and idx < start + 20000:
        cur = lines[idx].rstrip()
        nxt = lines[idx + 1].rstrip() if idx + 1 < len(lines) else ''
        if not cur.strip():
            blank_run += 1
            if blank_run >= 3 and content_lines:
                break
            total += len(lines[idx]) + 1
            idx += 1
            continue
        blank_run = 0
        prose_start = (len(cur.strip()) >= 60 and len(nxt.strip()) >= 60
                       and not INDEX_LINE_RE.search(cur) and not INDEX_LINE_RE.search(nxt))
        if prose_start:
            break
        content_lines += 1
        index_like += 1 if INDEX_LINE_RE.search(cur) else 0
        if content_lines > 400 and index_like < 0.5 * content_lines:
            break
        total += len(lines[idx]) + 1
        idx += 1
    return total, idx


def span_chars(regex, text):
    return sum(m.end() - m.start() for m in regex.finditer(text))


def doc_features(text, examples, doc_id):
    """Per-document measurements. `examples` is filled with up to N_EXAMPLES snippets per issue."""
    feats = {}
    n_chars = len(text)

    def note(issue, start, end):
        lst = examples.setdefault(issue, [])
        if len(lst) < N_EXAMPLES:
            lst.append(f'{doc_id}: ' + text[max(0, start - 60):min(len(text), end + 60)][:290])

    # credits preamble: the paragraph containing a credit line in the first 3000 chars
    m = CREDIT_RE.search(text, 0, 3000)
    if m:
        para_start = text.rfind('\n\n', 0, m.start()) + 1 if text.rfind('\n\n', 0, m.start()) >= 0 else 0
        para_end = text.find('\n\n', m.end())
        para_end = n_chars if para_end < 0 else para_end
        feats['credits'] = para_end - para_start
        note('credits', m.start(), min(para_end, m.start() + 200))
    else:
        feats['credits'] = 0

    # A trailer is only counted in the final 3000 chars; earlier hits are prefaces such as "** End of PG Preface **".
    m = TRAILER_RE.search(text, max(0, n_chars - 3000))
    feats['trailer'] = n_chars - m.start() if m else 0
    feats['trailer_like_midtext'] = 1 if (not m and TRAILER_RE.search(text)) else 0
    if m:
        note('trailer', m.start(), m.start() + 150)

    pg_chars = 0
    etext_chars = 0
    lines = text.split('\n')
    allcaps_chars = 0
    for line in lines:
        if PG_LINE_RE.search(line):
            pg_chars += len(line) + 1
        if ETEXT_LINE_RE.search(line):
            etext_chars += len(line) + 1
        letters = [ch for ch in line if ch.isalpha()]
        if len(letters) >= 4 and all(ch.isupper() for ch in letters):
            allcaps_chars += len(line) + 1
    feats['pg_lines'] = pg_chars
    feats['etext_lines'] = etext_chars
    feats['allcaps_lines'] = allcaps_chars
    m = PG_LINE_RE.search(text)
    if m:
        note('pg_lines', m.start(), m.end() + 100)

    legal = list(LEGAL_RE.finditer(text))
    if legal:
        start, end = legal[0].start(), legal[-1].end()
        feats['legal'] = end - start + 1 if end - start < 40000 else len(legal[0].group())
        note('legal', start, start + 200)
    else:
        feats['legal'] = 0

    feats['html_note'] = span_chars(HTML_NOTE_RE, text)
    m = HTML_NOTE_RE.search(text)
    if m:
        note('html_note', m.start(), m.start() + 150)

    tn = 0
    covered_until = -1
    for m in TN_BRACKET_RE.finditer(text):
        tn += m.end() - m.start()
        covered_until = max(covered_until, m.end())
        note('transcriber_note', m.start(), m.start() + 200)
    for m in TN_PARA_RE.finditer(text):
        if m.start() < covered_until:
            continue
        gap = re.search(r'\n[ \t]*\n[ \t]*\n', text[m.start():m.start() + 6000])
        length = gap.start() if gap else min(6000, n_chars - m.start())
        tn += length
        note('transcriber_note', m.start(), m.start() + 200)
    feats['transcriber_note'] = tn
    feats['transcriber_any'] = 1 if TN_ANY_RE.search(text) else 0

    for name, regex in (('illustration', ILLUS_RE), ('footnote_block', FOOTNOTE_BLOCK_RE), ('page_marker', PAGE_RE),
                        ('html_tag', HTML_TAG_RE), ('mojibake', MOJIBAKE_RE), ('asterism', ASTERISM_RE),
                        ('diacritic_notation', DIACRITIC_NOTE_RE), ('trailing_number_line', VERSE_LINENO_RE)):
        total = 0
        for m in regex.finditer(text):
            total += m.end() - m.start()
            if total == m.end() - m.start():
                note(name, m.start(), m.end())
        feats[name] = total
    feats['fn_marker_count'] = len(FN_MARK_RE.findall(text))
    italics = ITALIC_RE.findall(text)
    feats['italic_spans'] = len(italics)
    feats['bold_spans'] = len(BOLD_RE.findall(text))
    feats['speaker_lines'] = len(SPEAKER_RE.findall(text))
    feats['stage_dirs'] = len(STAGE_RE.findall(text))
    feats['blank_excess'] = sum(len(m.group()) - 2 for m in BLANKRUN_RE.finditer(text))
    feats['header_fragment'] = 1 if HEADER_FRAG_RE.search(text) else 0
    if feats['header_fragment']:
        note('header_fragment', 0, 80)

    box = 0
    for line in lines:
        stripped = line.strip()
        if len(stripped) >= 5 and not ASTERISM_RE.match(line):
            sym = sum(1 for ch in stripped if ch in '|+-=_~#:.')
            if sym / len(stripped) >= 0.6 or stripped.count('|') >= 2:
                box += len(line) + 1
    feats['ascii_art_table'] = box

    toc = 0
    covered_until = -1
    for idx, line in enumerate(lines):
        if idx > covered_until and CONTENTS_HEAD_RE.match(line):
            block_chars, block_end = block_after_heading(lines, idx)
            toc += block_chars
            covered_until = block_end
    feats['toc_index_block'] = toc
    feats['index_lines'] = sum(len(ln) + 1 for ln in lines if len(ln) < 100 and ln.strip() and INDEX_LINE_RE.search(ln))

    # hard wrap and verse
    wrap_newlines = 0
    multi_line_lengths = []
    verse_lines = 0
    nonempty = 0
    for plines in paragraphs_with_lines(text):
        nonempty += len(plines)
        if len(plines) >= 2:
            wrap_newlines += len(plines) - 1
            multi_line_lengths.extend(len(ln) for ln in plines[:-1])
            body = plines[:-1] if len(plines) > 2 else plines
            if sum(1 for ln in body if len(ln.strip()) < 50) >= 0.8 * len(body) and len(plines) >= 3:
                verse_lines += len(plines)
    if multi_line_lengths:
        in_band = sum(1 for ln in multi_line_lengths if 50 <= ln <= 80)
        feats['wrap_band_frac'] = in_band / len(multi_line_lengths)
        feats['wrap_median'] = sorted(multi_line_lengths)[len(multi_line_lengths) // 2]
    else:
        feats['wrap_band_frac'] = 0.0
        feats['wrap_median'] = 0
    feats['wrap_newlines'] = wrap_newlines
    feats['verse_line_frac'] = verse_lines / nonempty if nonempty else 0.0
    feats['nonempty_lines'] = nonempty

    # stubs and catalog-like docs
    feats['digit_frac'] = sum(1 for ch in text if ch.isdigit()) / n_chars if n_chars else 0.0
    feats['stub_media'] = 1 if (n_chars < 5000 and STUB_MEDIA_RE.search(text)) else 0
    tail = text[int(n_chars * 0.85):]
    feats['ad_lines_tail'] = len(AD_RE.findall(tail))
    feats['ad_heading'] = 1 if AD_HEAD_RE.search(text) else 0

    # archaic spelling proxies
    lower_words = [w.lower() for w in WORD_RE.findall(text)]
    n_words = len(lower_words)
    archaic = sum(1 for w in lower_words if w in ARCHAIC)
    eth = sum(1 for w in lower_words if w.endswith('eth') and len(w) > 4 and not w.endswith('ieth') and w not in ETH_EXCLUDE)
    elision = len(ELISION_RE.findall(text.lower()))
    feats['archaic_per_1k'] = 1000 * (archaic + eth + elision) / n_words if n_words else 0.0
    feats['long_s_thorn_yogh'] = sum(text.count(ch) for ch in 'ſþȝÞȜ')
    return feats


def chunk_languages(text, chunk=4000, max_chunks=60):
    if len(text) <= chunk:
        return [classify_language(text)[0]]
    n = len(text) // chunk
    step = max(1, n // max_chunks)
    return [classify_language(text[i * chunk:(i + 1) * chunk])[0] for i in range(0, n, step)]


def run_index():
    for shard, path in RAW_SHARDS.items():
        with open(os.path.join(WORK, f'idx_raw_{shard}.jsonl'), 'w') as out:
            for doc in iter_docs(path):
                out.write(json.dumps({'id': doc['id'], 'len': len(doc['text']), 'meta': doc['metadata']}) + '\n')


def stats_worker(task):
    kind, shard = task
    path = RAW_SHARDS[shard] if kind == 'raw' else FILT_SHARDS[shard]
    out_path = os.path.join(WORK, f'stats_{kind}_{shard}.jsonl')
    examples = {}
    id_range = None
    if kind == 'filt':
        raw_shard = next(r for r, f in FILT_FOR_RAW.items() if f == shard)
        raw_ids = [json.loads(l)['id'] for l in open(os.path.join(WORK, f'idx_raw_{raw_shard}.jsonl'))]
        id_range = (min(raw_ids), max(raw_ids))
    with open(out_path, 'w') as out:
        for doc in iter_docs(path):
            if id_range and not (id_range[0] <= doc['id'] <= id_range[1]):
                continue
            text = doc['text']
            meta = doc['metadata']
            rec = {'id': doc['id'], 'kind': kind, 'shard': shard, 'meta_lang': meta.get('language'),
                   'license': meta.get('license'), 'title': (meta.get('title') or '')[:200],
                   'url_form': 'files' if '/files/' in (meta.get('url') or '') else 'ebooks',
                   'chars': len(text), 'words': len(text.split())}
            lang, en_ratio, non_latin = classify_language(text[:200000])
            rec.update(lang=lang, en_stop_ratio=round(en_ratio, 4), non_latin=round(non_latin, 4))
            chunks = chunk_languages(text)
            rec['chunks'] = len(chunks)
            rec['chunks_non_en'] = sum(1 for c in chunks if c not in ('en',))
            rec['chunks_foreign'] = sum(1 for c in chunks if c not in ('en', 'unknown', 'empty'))
            rec.update(doc_features(text, examples, doc['id']))
            out.write(json.dumps(rec) + '\n')
    with open(os.path.join(WORK, f'examples_{kind}_{shard}.json'), 'w') as handle:
        json.dump(examples, handle, ensure_ascii=False, indent=1)
    return out_path


def run_stats():
    tasks = [('raw', s) for s in RAW_SHARDS] + [('filt', s) for s in FILT_SHARDS]
    with multiprocessing.Pool(2) as pool:
        for path in pool.imap_unordered(stats_worker, tasks):
            print('done', path, flush=True)


# ---------------------------------------------------------------- duplicates
SHINGLE = 5
SAMPLE_MOD = 64          # keep 1/64 of 5-shingle hashes (mod-p sampling, Broder 1997)
MAX_POSTING = 50         # skip shingle hashes shared by >50 docs (stock phrases); pair counts are then lower bounds
POWERS = [np.uint64(pow(1000003, k, 2 ** 64)) for k in range(SHINGLE)]


def normalize_for_exact(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()


def shingle_sample(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < SHINGLE:
        return np.zeros(0, dtype=np.uint64)
    hashes = np.fromiter(map(xxhash.xxh64_intdigest, words), dtype=np.uint64, count=len(words))
    combined = np.zeros(len(words) - SHINGLE + 1, dtype=np.uint64)
    for k in range(SHINGLE):
        combined += hashes[k:len(words) - SHINGLE + 1 + k] * POWERS[k]
    combined ^= combined >> np.uint64(29)
    combined *= np.uint64(0xbf58476d1ce4e5b9)
    combined ^= combined >> np.uint64(32)
    return np.unique(combined[combined % np.uint64(SAMPLE_MOD) == 0])


def run_dedup():
    ids, sizes, exact = [], [], collections.defaultdict(list)
    all_hashes, all_docs = [], []
    for shard, path in RAW_SHARDS.items():
        for doc in iter_docs(path):
            text = doc['text']
            idx = len(ids)
            ids.append(doc['id'])
            exact[hashlib.md5(normalize_for_exact(text).encode()).hexdigest()].append(doc['id'])
            sample = shingle_sample(text)
            sizes.append(len(sample))
            all_hashes.append(sample)
            all_docs.append(np.full(len(sample), idx, dtype=np.int32))
        print('dedup hashed', shard, len(ids), flush=True)
    hashes = np.concatenate(all_hashes)
    docs = np.concatenate(all_docs)
    order = np.argsort(hashes, kind='stable')
    hashes, docs = hashes[order], docs[order]
    boundaries = np.flatnonzero(np.diff(hashes)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(hashes)]])
    pair_counts = collections.Counter()
    skipped = 0
    for start, end in zip(starts, ends):
        n = end - start
        if n < 2:
            continue
        if n > MAX_POSTING:
            skipped += 1
            continue
        members = sorted(set(docs[start:end].tolist()))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair_counts[(members[i], members[j])] += 1
    pairs = []
    for (a, b), inter in pair_counts.items():
        if inter < 5:
            continue
        union = sizes[a] + sizes[b] - inter
        pairs.append({'a': ids[a], 'b': ids[b], 'inter': inter, 'size_a': sizes[a], 'size_b': sizes[b],
                      'jaccard': inter / union, 'cont_a_in_b': inter / sizes[a], 'cont_b_in_a': inter / sizes[b]})
    result = {'n_docs': len(ids), 'sizes': dict(zip(ids, sizes)), 'skipped_common_hashes': skipped,
              'exact_groups': [g for g in exact.values() if len(g) > 1], 'pairs': pairs}
    with open(os.path.join(WORK, 'dedup.json'), 'w') as handle:
        json.dump(result, handle)
    print('dedup pairs', len(pairs), 'exact groups', len(result['exact_groups']), flush=True)


# ---------------------------------------------------------------- raw vs filtered
def run_filtdiff():
    out = {}
    for raw_shard, filt_shard in FILT_FOR_RAW.items():
        # Both shards are sorted by id string, so a streaming merge join keeps memory at one document per side.
        raw_iter = iter_docs(RAW_SHARDS[raw_shard])
        filt_iter = iter_docs(FILT_SHARDS[filt_shard])
        raw_ids = []
        extra_in_filtered = []
        pending = next(filt_iter, None)
        prev_filt_id = ''
        dropped, identical, modified = [], 0, []
        removed_lines = collections.Counter()
        removed_line_docs = collections.Counter()
        added_nonempty = 0
        raw_chars_kept = 0
        removed_chars = 0
        chapter3 = collections.Counter()
        for doc in raw_iter:
            text = doc['text']
            raw_ids.append(doc['id'])
            assert len(raw_ids) < 2 or raw_ids[-2] < raw_ids[-1], 'raw shard not sorted by id string'
            while pending is not None and pending['id'] < doc['id']:
                if raw_ids[0] <= pending['id']:
                    extra_in_filtered.append(pending['id'])
                assert prev_filt_id <= pending['id'], 'filtered shard not sorted by id string'
                prev_filt_id = pending['id']
                pending = next(filt_iter, None)
            if pending is None or pending['id'] != doc['id']:
                dropped.append(doc['id'])
                continue
            ftext = pending['text']
            prev_filt_id = pending['id']
            pending = next(filt_iter, None)
            raw_chars_kept += len(text)
            if ftext == text:
                identical += 1
                continue
            removed_chars += len(text) - len(ftext)
            a = collections.Counter(text.split('\n'))
            b = collections.Counter(ftext.split('\n'))
            diff = a - b
            added = b - a
            added_nonempty += sum(v for k, v in added.items() if k.strip())
            removed_lines.update(diff)
            removed_line_docs.update(diff.keys())
            # does any removed line string survive elsewhere in the same filtered doc?
            survives = sum(1 for k in diff if b[k] > 0)
            modified.append({'id': doc['id'], 'raw_chars': len(text), 'removed_chars': len(text) - len(ftext),
                             'removed_line_kinds': len(diff), 'removed_line_count': sum(diff.values()),
                             'removed_kinds_still_present': survives})
            has = lambda t, s: re.search(rf'(?m)^[ \t]*{s}\.?[ \t]*$', t) is not None
            if has(text, 'CHAPTER III'):
                chapter3['raw_has_III'] += 1
                if not has(ftext, 'CHAPTER III'):
                    chapter3['filt_lost_III'] += 1
                    if has(ftext, 'CHAPTER II') and has(ftext, 'CHAPTER IV'):
                        chapter3['filt_lost_III_kept_II_and_IV'] += 1
            for numeral in ('CHAPTER I', 'CHAPTER II', 'CHAPTER IV', 'CHAPTER V'):
                if has(text, numeral) and not has(ftext, numeral):
                    chapter3[f'filt_lost_{numeral}'] += 1
        out[raw_shard] = {
            'filtered_shard': filt_shard, 'raw_docs': len(raw_ids), 'kept': len(raw_ids) - len(dropped),
            'dropped': dropped, 'identical': identical, 'modified': modified,
            'filtered_ids_not_in_raw': extra_in_filtered, 'raw_chars_kept_docs': raw_chars_kept,
            'removed_chars_kept_docs': removed_chars, 'added_nonempty_lines': added_nonempty,
            'top_removed_lines': [(k, v, removed_line_docs[k]) for k, v in removed_lines.most_common(80)],
            'distinct_removed_lines': len(removed_lines), 'total_removed_lines': sum(removed_lines.values()),
            'chapter_heading_check': dict(chapter3)}
        print('filtdiff', raw_shard, out[raw_shard]['raw_docs'], len(dropped), identical, len(modified), flush=True)
    with open(os.path.join(WORK, 'filtdiff.json'), 'w') as handle:
        json.dump(out, handle, ensure_ascii=False)


# ---------------------------------------------------------------- report
def quantiles(values, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)):
    arr = np.array(sorted(values))
    return {f'p{int(q * 100)}' if q < 1 else 'max': float(np.quantile(arr, q)) for q in qs}


def load_stats(kind, shards):
    rows = []
    for shard in shards:
        with open(os.path.join(WORK, f'stats_{kind}_{shard}.jsonl')) as handle:
            rows.extend(json.loads(l) for l in handle)
    return rows


CHAR_ISSUES = ['credits', 'trailer', 'pg_lines', 'etext_lines', 'legal', 'html_note', 'transcriber_note', 'illustration',
               'footnote_block', 'page_marker', 'html_tag', 'mojibake', 'asterism', 'diacritic_notation', 'trailing_number_line',
               'ascii_art_table', 'toc_index_block', 'index_lines', 'blank_excess', 'allcaps_lines']


def issue_table(rows):
    total_chars = sum(r['chars'] for r in rows)
    table = {}
    for issue in CHAR_ISSUES:
        affected = [r for r in rows if r[issue] > 0]
        table[issue] = {'doc_frac': round(len(affected) / len(rows), 4),
                        'char_frac': round(sum(r[issue] for r in rows) / total_chars, 5),
                        'docs': len(affected)}
    return table


def run_report():
    raw = load_stats('raw', RAW_SHARDS)
    filt_all = load_stats('filt', FILT_SHARDS)
    report = {'n_raw_docs': len(raw), 'raw_total_chars': sum(r['chars'] for r in raw),
              'raw_total_words': sum(r['words'] for r in raw)}
    report['chars_quantiles'] = quantiles([r['chars'] for r in raw])
    report['words_quantiles'] = quantiles([r['words'] for r in raw])
    report['license_counts'] = collections.Counter(r['license'] for r in raw)
    report['meta_lang_counts'] = collections.Counter(r['meta_lang'] for r in raw)
    report['url_form_counts'] = collections.Counter(r['url_form'] for r in raw)
    report['meta_lang_char_share_non_en'] = sum(r['chars'] for r in raw if r['meta_lang'] != 'en') / report['raw_total_chars']
    conf = collections.Counter((r['meta_lang'] if r['meta_lang'] in STOP or r['meta_lang'] in ('en',) else 'other', r['lang']) for r in raw)
    report['lang_confusion_meta_vs_heuristic'] = {f'{a}->{b}': v for (a, b), v in conf.most_common()}
    en_meta = [r for r in raw if r['meta_lang'] == 'en']
    report['en_meta_heuristic_not_en'] = [(r['id'], r['lang'], r['en_stop_ratio'], r['title'][:60]) for r in en_meta if r['lang'] != 'en'][:40]
    report['en_meta_heuristic_not_en_count'] = sum(1 for r in en_meta if r['lang'] != 'en')
    mixed = [r for r in en_meta if r['chunks'] and r['chunks_foreign'] / r['chunks'] >= 0.1]
    report['en_meta_docs_with_ge10pct_foreign_chunks'] = {
        'docs': len(mixed), 'doc_frac_of_en': len(mixed) / len(en_meta),
        'est_char_frac_foreign_in_en': sum(r['chars'] * r['chunks_foreign'] / r['chunks'] for r in en_meta if r['chunks']) / sum(r['chars'] for r in en_meta),
        'examples': [(r['id'], r['chunks_foreign'], r['chunks'], r['title'][:60]) for r in mixed[:15]]}
    report['issues_raw_all'] = issue_table(raw)
    report['issues_raw_en_meta'] = issue_table(en_meta)
    # filtered: restrict to ids that are in the compared raw shards
    raw_ids_compared = {r['id'] for r in raw if r['shard'] in FILT_FOR_RAW}
    filt = [r for r in filt_all if r['id'] in raw_ids_compared]
    report['n_filtered_matched_docs'] = len(filt)
    report['issues_filtered_matched'] = issue_table(filt)
    filt_ids = {r['id'] for r in filt}
    report['issues_raw_same_ids_as_filtered'] = issue_table([r for r in raw if r['id'] in filt_ids])
    report['filtered_heuristic_lang'] = collections.Counter(r['lang'] for r in filt)
    report['filtered_meta_lang'] = collections.Counter(r['meta_lang'] for r in filt)

    def frac(pred, rows):
        return round(sum(1 for r in rows if pred(r)) / len(rows), 4)
    hard = lambda r: r['wrap_band_frac'] >= 0.5
    report['hardwrap'] = {'doc_frac_ge50pct_lines_50_80': frac(hard, raw),
                          'wrap_newlines_per_1k_chars': 1000 * sum(r['wrap_newlines'] for r in raw) / report['raw_total_chars'],
                          'wrap_median_quantiles': quantiles([r['wrap_median'] for r in raw if r['wrap_median']])}
    report['verse'] = {'doc_frac_verse_ge30pct': frac(lambda r: r['verse_line_frac'] >= 0.3, raw),
                       'doc_frac_verse_ge60pct': frac(lambda r: r['verse_line_frac'] >= 0.6, raw),
                       'verse_and_hardwrap_both': frac(lambda r: r['verse_line_frac'] >= 0.3 and hard(r), raw)}
    report['plays'] = {'doc_frac_speaker_ge50': frac(lambda r: r['speaker_lines'] >= 50, raw),
                       'doc_frac_stage_dirs_ge5': frac(lambda r: r['stage_dirs'] >= 5, raw)}
    report['markup'] = {'doc_frac_italic_underscore': frac(lambda r: r['italic_spans'] > 0, raw),
                        'italic_underscore_chars_frac': 2 * sum(r['italic_spans'] for r in raw) / report['raw_total_chars'],
                        'doc_frac_bold_equals_ge3': frac(lambda r: r['bold_spans'] >= 3, raw),
                        'doc_frac_fn_markers_ge5': frac(lambda r: r['fn_marker_count'] >= 5, raw)}
    report['stubs'] = {'lt1000_chars': frac(lambda r: r['chars'] < 1000, raw), 'lt5000_chars': frac(lambda r: r['chars'] < 5000, raw),
                       'stub_media': frac(lambda r: r['stub_media'] == 1, raw),
                       'stub_examples': [(r['id'], r['chars'], r['title'][:60]) for r in raw if r['chars'] < 1000][:20]}
    report['digit_heavy'] = {'doc_frac_digit_gt_0.05': frac(lambda r: r['digit_frac'] > 0.05, raw),
                             'doc_frac_digit_gt_0.10': frac(lambda r: r['digit_frac'] > 0.10, raw),
                             'char_frac_docs_digit_gt_0.10': sum(r['chars'] for r in raw if r['digit_frac'] > 0.10) / report['raw_total_chars'],
                             'examples': [(r['id'], round(r['digit_frac'], 3), r['title'][:60]) for r in sorted(raw, key=lambda r: -r['digit_frac'])[:12]]}
    report['ads'] = {'doc_frac_tail_ad_lines_ge5': frac(lambda r: r['ad_lines_tail'] >= 5, raw),
                     'doc_frac_ad_heading': frac(lambda r: r['ad_heading'] == 1, raw)}
    report['archaic'] = {'per_1k_quantiles': quantiles([r['archaic_per_1k'] for r in en_meta]),
                         'en_doc_frac_gt5_per_1k': frac(lambda r: r['archaic_per_1k'] > 5, en_meta),
                         'en_doc_frac_gt20_per_1k': frac(lambda r: r['archaic_per_1k'] > 20, en_meta),
                         'doc_frac_long_s_thorn_yogh_ge10': frac(lambda r: r['long_s_thorn_yogh'] >= 10, en_meta)}
    report['header_fragment_doc_frac'] = frac(lambda r: r['header_fragment'] == 1, raw)
    report['trailer_like_midtext_docs'] = sum(r['trailer_like_midtext'] for r in raw)
    report['transcriber_any_doc_frac'] = frac(lambda r: r['transcriber_any'] == 1, raw)
    titles = [r['title'] for r in raw]
    report['title_patterns'] = {
        'complete_omnibus': sum(1 for t in titles if re.search(r'(?i)\b(complete|omnibus|entire collection|collected works)\b', t)),
        'part_volume_number': sum(1 for t in titles if re.search(r'(?i)\b(part|volume|vol\.?|book|no\.)\s*[\divxlc]+\b', t))}

    # duplicates
    dd = json.load(open(os.path.join(WORK, 'dedup.json')))
    by_id = {r['id']: r for r in raw}
    report['exact_dup_groups'] = [g for g in dd['exact_groups']]
    report['exact_dup_extra_docs'] = sum(len(g) - 1 for g in dd['exact_groups'])
    reliable = {i for i, s in dd['sizes'].items() if s >= 20}
    near = [p for p in dd['pairs'] if p['jaccard'] >= 0.8 and p['a'] in reliable and p['b'] in reliable]
    contained = {}
    for p in dd['pairs']:
        for small, big, cont in ((p['a'], p['b'], p['cont_a_in_b']), (p['b'], p['a'], p['cont_b_in_a'])):
            if small in reliable and cont >= 0.8 and dd['sizes'][small] <= dd['sizes'][big]:
                if cont > contained.get(small, (None, 0))[1]:
                    contained[small] = (big, cont)
    near_docs = {p['a'] for p in near} | {p['b'] for p in near}
    report['near_dup'] = {
        'reliable_docs(>=20 sampled shingles)': len(reliable), 'skipped_common_hashes': dd['skipped_common_hashes'],
        'pairs_jaccard_ge_0.8': len(near), 'docs_in_such_pairs': len(near_docs),
        'pairs_jaccard_ge_0.5': sum(1 for p in dd['pairs'] if p['jaccard'] >= 0.5 and p['a'] in reliable and p['b'] in reliable),
        'docs_contained_ge_0.8_in_other_doc': len(contained),
        'contained_doc_frac': len(contained) / len(raw),
        'contained_char_frac': sum(by_id[i]['chars'] for i in contained) / report['raw_total_chars'],
        'near_examples': [(p['a'], by_id[p['a']]['title'][:50], p['b'], by_id[p['b']]['title'][:50], round(p['jaccard'], 3)) for p in sorted(near, key=lambda p: -p['jaccard'])[:12]],
        'contained_examples': [(s, by_id[s]['title'][:50], b, by_id[b]['title'][:50], round(c, 3)) for s, (b, c) in list(contained.items())[:15]]}

    # raw vs filtered
    fd = json.load(open(os.path.join(WORK, 'filtdiff.json')))
    fsum = {}
    for shard, info in fd.items():
        dropped = info['dropped']
        drop_rows = [by_id[i] for i in dropped]
        drop_en = [r for r in drop_rows if r['meta_lang'] == 'en']
        kept_rows = [by_id[i] for i in by_id if by_id[i]['shard'] == shard and i not in set(dropped)]
        complete_re = re.compile(r'(?i)\b(complete|omnibus|entire collection)\b')
        dropped_en_contained = [r['id'] for r in drop_en if r['id'] in contained]
        fsum[shard] = {
            'filtered_shard': info['filtered_shard'], 'raw_docs': info['raw_docs'], 'kept': info['kept'], 'dropped': len(dropped),
            'dropped_char_frac': sum(r['chars'] for r in drop_rows) / sum(by_id[i]['chars'] for i in by_id if by_id[i]['shard'] == shard),
            'dropped_meta_lang': collections.Counter(r['meta_lang'] for r in drop_rows),
            'kept_meta_lang': collections.Counter(r['meta_lang'] for r in kept_rows),
            'dropped_en': len(drop_en),
            'dropped_en_title_complete_omnibus': sum(1 for r in drop_en if complete_re.search(r['title'])),
            'kept_en_title_complete_omnibus': sum(1 for r in kept_rows if r['meta_lang'] == 'en' and complete_re.search(r['title'])),
            'dropped_en_contained_ge0.8_in_sample_doc': len(dropped_en_contained),
            'kept_en_contained_ge0.8_in_sample_doc': sum(1 for r in kept_rows if r['meta_lang'] == 'en' and r['id'] in contained),
            'dropped_en_digit_gt_0.05': sum(1 for r in drop_en if r['digit_frac'] > 0.05),
            'kept_en_digit_gt_0.05': sum(1 for r in kept_rows if r['meta_lang'] == 'en' and r['digit_frac'] > 0.05),
            'dropped_en_median_chars': float(np.median([r['chars'] for r in drop_en])) if drop_en else None,
            'kept_en_median_chars': float(np.median([r['chars'] for r in kept_rows if r['meta_lang'] == 'en'])),
            'dropped_en_examples': [(r['id'], r['chars'], r['title'][:60]) for r in drop_en[:25]],
            'identical_text': info['identical'], 'modified_text': len(info['modified']),
            'removed_char_frac_of_kept': info['removed_chars_kept_docs'] / info['raw_chars_kept_docs'],
            'modified_doc_removed_char_frac_quantiles': quantiles([m['removed_chars'] / m['raw_chars'] for m in info['modified']]) if info['modified'] else None,
            'removed_line_kinds_still_present_anywhere': sum(m['removed_kinds_still_present'] for m in info['modified']),
            'added_nonempty_lines': info['added_nonempty_lines'],
            'distinct_removed_lines': info['distinct_removed_lines'], 'total_removed_lines': info['total_removed_lines'],
            'top_removed_lines': info['top_removed_lines'][:30],
            'filtered_ids_not_in_raw': len(info['filtered_ids_not_in_raw']),
            'chapter_heading_check': info['chapter_heading_check']}
    report['filtered_comparison'] = fsum

    examples = {}
    for name in os.listdir(WORK):
        if name.startswith('examples_raw_'):
            for issue, lst in json.load(open(os.path.join(WORK, name))).items():
                examples.setdefault(issue, []).extend(lst[:2])
    report['examples'] = examples
    with open(os.path.join(WORK, 'report.json'), 'w') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    os.makedirs(WORK, exist_ok=True)
    if step in ('index', 'all'):
        run_index()
    if step in ('stats', 'all'):
        run_stats()
    if step in ('dedup', 'all'):
        run_dedup()
    if step in ('filtdiff', 'all'):
        run_filtdiff()
    if step in ('report', 'all'):
        run_report()

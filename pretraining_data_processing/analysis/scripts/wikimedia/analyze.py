"""Quality analysis of a stratified sample of common-pile/wikimedia (raw) against common-pile/wikimedia_filtered.

Usage: python3 -I analyze.py WORKDIR
WORKDIR/raw/*.jsonl.gz are raw shards, WORKDIR/filtered/*.json.gz are filtered shards.
Outputs go to WORKDIR/out/.
"""
import collections
import gzip
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import re
import sys
import time
import zlib

import numpy as np
import xxhash

WORKDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(WORKDIR, 'out')
SEED = 1234
MINHASH_PER_PROJECT = 8000
MINHASH_PERMS = 128
LSH_BANDS, LSH_ROWS = 16, 8
NEAR_DUP_THRESHOLD = 0.8
EXAMPLES_PER_ISSUE = 6

# Namespace numbers are MediaWiki defaults; odd numbers are talk namespaces.
NS_NAMES = {'0': 'main', '1': 'talk', '2': 'user', '3': 'user_talk', '4': 'project', '5': 'project_talk',
            '6': 'file', '7': 'file_talk', '8': 'mediawiki', '10': 'template', '11': 'template_talk',
            '12': 'help', '14': 'category', '15': 'category_talk', '100': 'ns100', '102': 'ns102',
            '104': 'ns104', '106': 'ns106', '828': 'module'}

ENGLISH_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also its may one which""".split())

# Function words of French, German, Spanish, Italian, Portuguese, Dutch and Latin that are not English words.
FOREIGN_STOPWORDS = frozenset("""
le la les des du et est une pour dans qui sur pas au aux avec ce cette sont par il elle ne se nous vous leur
der das und ist nicht ein eine zu den von mit sich auf für im dem auch es wird werden sie ich wir ihr oder
el los las del y es un una por con para que su al lo como más pero sus ya muy está
il di che è della delle degli gli nel nella sono non si da anche questo
os da do das dos em um uma com não é na no ao pelo pela
het een van en dat niet op te zijn voor ook maar wordt
et est non ad cum quod quae sed ut esse enim autem atque ab ex""".split()) - ENGLISH_STOPWORDS

# Markup debris: each entry is (name, regex). Character coverage is the union of match spans.
MARKUP_PATTERNS = [
    ('template_braces', r'\{\{[^{}\n]{0,300}\}\}|\{\{|\}\}'),
    ('wikilink_brackets', r'\[\[[^\[\]\n]{0,300}\]\]|\[\[|\]\]'),
    ('file_image_syntax', r'(?i)\[\[(?:File|Image|Media):[^\n]*|\|\s*(?:thumb|thumbnail|frameless|upright(?:=[\d.]+)?|\d+(?:x\d+)?px)\s*(?=\||\]\])'),
    ('bracket_extlink', r'\[https?://[^\]\s]+(?: [^\]\n]{0,200})?\]'),
    ('bold_italic_quotes', r"'{2,}"),
    ('html_tag', r'(?i)</?(?:ref|li|ol|ul|div|span|br|sup|sub|small|big|center|font|table|tr|td|th|p|b|i|u|s|code|pre|nowiki|gallery|blockquote|references|poem|score|syntaxhighlight|source|math|chem|timeline|includeonly|noinclude|onlyinclude|abbr|em|strong|hr|tt|del|ins|mapframe|templatestyles|imagemap|hiero|section|cite|dl|dd|dt|caption|pages|onlyinclude)\b[^<>\n]{0,200}>'),
    ('html_entity', r'&(?:[a-zA-Z]{2,8}|#\d{2,6}|#x[0-9a-fA-F]{2,5});'),
    ('table_syntax', r'(?m)^[ \t]*(?:\{\||\|\}|\|-|\|\+|! ).*$|\b(?:style|class|cellpadding|cellspacing|border|align|width|colspan|rowspan|bgcolor|valign)\s*=\s*"[^"\n]*"'),
    ('magic_word', r'__(?:NOTOC|TOC|FORCETOC|NOEDITSECTION|DISAMBIG|NOINDEX|INDEX|NEWSECTIONLINK|NOGALLERY|HIDDENCAT|STATICREDIRECT|NOTITLECONVERT|NOCONTENTCONVERT)__'),
    ('category_line', r'(?m)^\s*Category:.*$'),
]
MARKUP_RES = [(name, re.compile(pat)) for name, pat in MARKUP_PATTERNS]

# Holes left where the converter dropped a template or link label: each is counted per line.
HOLE_PATTERNS = [
    ('function_word_then_punct', r'(?i)\b(?:the|an|of|from|with|between|into|than)\s+[,.;:)]'),
    ('empty_or_comma_paren', r'\(\s*[,;](?:[\s,;]*)|\(\s*\)'),
    ('double_comma', r',[ \t]*,'),
    ('double_space_in_sentence', r'(?<=[a-z,])[ \t]{2,}(?=[a-z])'),
    ('space_before_punct', r'(?<=[a-z]) [,.;:](?=[ \t\n]|$)'),
]
HOLE_RES = [(name, re.compile(pat)) for name, pat in HOLE_PATTERNS]

EMPTY_ITEM_RE = re.compile(r'(?m)^ ?\*[ \t]*(?:[^\n]{1,60}?:[ \t,;]*)?$')
BLANK_RUN_RE = re.compile(r'\n{4,}')
SIGNATURE_RE = re.compile(r'\d{1,2}:\d{2}, \d{1,2} [A-Z][a-z]+ \d{4} \(UTC\)')
DISAMBIG_TEXT_RE = re.compile(r'__DISAMBIG__|\b(?:may|can|might|could) (?:also )?(?:refer|stand) (?:to|for):|There (?:is|are) more than one (?:place|article|page|person)s? (?:called|named|titled)')
REF_HEADING_RE = re.compile(r'(?mi)^(?:references|external links|see also|further reading|notes|bibliography|sources|citations|footnotes|works cited|notes and references|general and cited references)[ \t]*$')
LIST_LINE_RE = re.compile(r'^\s*(?:\*|#|\d+[.)]\s|\|)')
URL_RE = re.compile(r'https?://\S+|\bwww\.[a-z0-9-]+\.[a-z]{2,}\S*')
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
IPV4_RE = re.compile(r'\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b')
LATEX_DISPLAY_RE = re.compile(r'\$\$.+?\$\$', re.S)
LATEX_SOURCE_RE = re.compile(r'\\documentclass|\\begin\{document\}|\\usepackage')
FLAG_EMOJI_RE = re.compile(r'[\U0001F1E6-\U0001F1FF]{2}')
TRANSLATION_LINE_RE = re.compile(r"(?m)^ ?\* [A-Z][A-Za-z' ()-]{1,30}:(?: [^\n]*)?$")
CODE_TITLE_RE = re.compile(r'\.(?:js|css|json|lua|py)$')
WORD_RE = re.compile(r'[^\W\d_]+')
PAGE_NUMBER_LINE_RE = re.compile(r'(?m)^[ \t]*\[?\d{1,4}[\].]?[ \t]*$')
SHINGLE_TOKEN_RE = re.compile(r'\w+')

ISSUE_BITS = ['non_main_ns', 'talk_ns', 'user_ns', 'title_only', 'body_lt20w', 'body_lt50w', 'disambig',
              'markup_any', 'holes_any', 'holes_dense', 'empty_items', 'blank_run', 'signature', 'ref_heading',
              'list_only', 'dup_lines_gt20pct', 'code_page', 'latex_source', 'non_english', 'low_stopword', 'lang_undetermined',
              'url', 'email', 'ipv4', 'flag_emoji', 'starts_lowercase', 'ends_unterminated', 'page_number_line']
BIT = {name: 1 << i for i, name in enumerate(ISSUE_BITS)}


def project_of(path):
    return os.path.basename(path).split('_', 1)[1].split('.')[0]


def project_from_source(source):
    return source.rsplit('/', 1)[-1].split('.')[0]


def load_meta(doc):
    meta = doc['metadata']
    return meta if isinstance(meta, dict) else eval_meta(meta)


def eval_meta(text):
    import ast
    return ast.literal_eval(text)


def split_body(text, title):
    if text.startswith(title):
        return text[len(title):].strip()
    return text.strip()


def span_union_length(spans):
    total = 0
    end = -1
    for start, stop in sorted(spans):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def language_scores(body):
    """Return (word count, English stopword ratio, non-Latin letter fraction, foreign stopword count, English stopword count)."""
    words = WORD_RE.findall(body)
    letters = sum(len(w) for w in words)
    if letters == 0:
        return 0, 0.0, 0.0, 0, 0
    non_latin = sum(1 for w in words for ch in w if ord(ch) > 0x24F)
    lowered = [w.lower() for w in words]
    stop = sum(1 for w in lowered if w in ENGLISH_STOPWORDS)
    foreign = sum(1 for w in lowered if w in FOREIGN_STOPWORDS)
    return len(words), stop / len(words), non_latin / letters, foreign, stop


def duplicate_line_char_fraction(body):
    lines = [ln.strip() for ln in body.split('\n')]
    lines = [ln for ln in lines if len(ln) >= 10]
    if len(lines) < 3:
        return 0.0
    counts = collections.Counter(lines)
    dup_chars = sum(len(ln) for ln in lines if counts[ln] > 1)
    return dup_chars / max(1, sum(len(ln) for ln in lines))


def normalize_for_hash(text):
    return ' '.join(text.lower().split())


def digest(text):
    return int.from_bytes(hashlib.blake2b(normalize_for_hash(text).encode('utf-8'), digest_size=8).digest(), 'little')


RECORD_DTYPE = np.dtype([('page', np.int64), ('ns', np.int32), ('chars', np.int64), ('body_words', np.int64),
                         ('flags', np.int64), ('digest', np.uint64), ('body_digest', np.uint64)])


def page_key(page, ns):
    # Page ids repeat across namespaces in these dumps only through the ns prefix, so the key keeps both.
    return np.asarray(page, dtype=np.int64) * 10000 + np.asarray(ns, dtype=np.int64)


def ns_int(ns):
    return int(ns) if ns.lstrip('-').isdigit() else -1


def analyze_doc(doc, meta):
    """Return (flags, per-issue char counts, per-issue hit details for examples)."""
    text = doc['text']
    title = meta.get('title', '')
    ns = meta.get('namespace', '?')
    body = split_body(text, title)
    flags = 0
    chars = {}
    spans_by_issue = {}

    if ns != '0':
        flags |= BIT['non_main_ns']
    if ns.isdigit() and int(ns) % 2 == 1:
        flags |= BIT['talk_ns']
    if ns in ('2', '3'):
        flags |= BIT['user_ns']
    body_words = len(body.split())
    if not body:
        flags |= BIT['title_only']
    if body_words < 20:
        flags |= BIT['body_lt20w']
    if body_words < 50:
        flags |= BIT['body_lt50w']
    if title.endswith('(disambiguation)') or DISAMBIG_TEXT_RE.search(text):
        flags |= BIT['disambig']

    markup_spans = []
    for name, regex in MARKUP_RES:
        spans = [m.span() for m in regex.finditer(text)]
        if spans:
            chars['markup:' + name] = span_union_length(spans)
            spans_by_issue['markup:' + name] = spans[0]
            markup_spans.extend(spans)
    if markup_spans:
        flags |= BIT['markup_any']
        chars['markup_any'] = span_union_length(markup_spans)

    hole_count = 0
    hole_line_chars = 0
    for line in body.split('\n'):
        line_holes = 0
        for name, regex in HOLE_RES:
            found = regex.findall(line)
            if found:
                line_holes += len(found)
                chars['holes:' + name] = chars.get('holes:' + name, 0) + len(found)
        if line_holes:
            hole_count += line_holes
            hole_line_chars += len(line)
    if hole_count:
        flags |= BIT['holes_any']
        chars['holes_any_line_chars'] = hole_line_chars
        chars['holes_count'] = hole_count
        if body_words and (hole_count * 100.0 / body_words) >= 1.0:
            flags |= BIT['holes_dense']
            chars['holes_dense_line_chars'] = hole_line_chars

    empty_items = [m.span() for m in EMPTY_ITEM_RE.finditer(body)]
    if empty_items:
        flags |= BIT['empty_items']
        chars['empty_items'] = span_union_length(empty_items)
    if BLANK_RUN_RE.search(body):
        flags |= BIT['blank_run']
    signatures = SIGNATURE_RE.findall(text)
    if signatures:
        flags |= BIT['signature']
        chars['signature_count'] = len(signatures)
    ref = REF_HEADING_RE.search(body)
    if ref:
        flags |= BIT['ref_heading']
        chars['ref_heading_to_end'] = len(body) - ref.start()

    lines = [ln for ln in body.split('\n') if ln.strip()]
    if len(lines) >= 3 and sum(1 for ln in lines if LIST_LINE_RE.match(ln)) >= 0.8 * len(lines):
        flags |= BIT['list_only']
    if duplicate_line_char_fraction(body) > 0.2:
        flags |= BIT['dup_lines_gt20pct']
    if ns in ('8', '828') or CODE_TITLE_RE.search(title):
        flags |= BIT['code_page']
    if LATEX_SOURCE_RE.search(body):
        flags |= BIT['latex_source']

    # Wikisource Page: namespace holds one scanned page, so a body that starts lower-case continues a sentence from the previous page.
    if body[:1].islower():
        flags |= BIT['starts_lowercase']
    if body and body_words >= 20 and body.rstrip()[-1:] not in '.!?"\'”’)]:;*':
        flags |= BIT['ends_unterminated']
    page_numbers = [m.span() for m in PAGE_NUMBER_LINE_RE.finditer(body)]
    if page_numbers:
        flags |= BIT['page_number_line']
        chars['page_number_line'] = span_union_length(page_numbers)
    n_words, stop_ratio, non_latin, foreign, english = language_scores(body)
    if n_words < 20:
        flags |= BIT['lang_undetermined']
    else:
        if non_latin > 0.5 or (foreign >= 5 and foreign > english):
            flags |= BIT['non_english']
        if stop_ratio < 0.10:
            flags |= BIT['low_stopword']

    for name, regex in (('url', URL_RE), ('email', EMAIL_RE), ('ipv4', IPV4_RE), ('flag_emoji', FLAG_EMOJI_RE)):
        found = [m.span() for m in regex.finditer(text)]
        if found:
            flags |= BIT[name]
            chars[name] = span_union_length(found)
            spans_by_issue[name] = found[0]
    latex = [m.span() for m in LATEX_DISPLAY_RE.finditer(text)]
    if latex:
        chars['latex_math'] = span_union_length(latex)
    translations = [m.span() for m in TRANSLATION_LINE_RE.finditer(body)]
    if translations:
        chars['lang_label_list_lines'] = span_union_length(translations)
    return flags, chars, spans_by_issue, body, body_words, stop_ratio, non_latin


def shingle_hashes(text):
    tokens = SHINGLE_TOKEN_RE.findall(text.lower())
    if len(tokens) < 5:
        return None
    grams = {' '.join(tokens[i:i + 5]) for i in range(len(tokens) - 4)}
    return np.fromiter((xxhash.xxh64_intdigest(g) for g in grams), dtype=np.uint64, count=len(grams))


def minhash_signature(hashes, mults, adds):
    sig = np.full(MINHASH_PERMS, np.iinfo(np.uint64).max, dtype=np.uint64)
    for start in range(0, len(hashes), 20000):
        chunk = hashes[start:start + 20000]
        with np.errstate(over='ignore'):
            values = chunk[None, :] * mults[:, None] + adds[:, None]
        np.minimum(sig, values.min(axis=1), out=sig)
    return sig


def analyze_shard(path):
    """Stream one raw shard; write compact per-doc records; return aggregate stats."""
    project = project_of(path)
    rng = random.Random(SEED + zlib.crc32(os.path.basename(path).encode()))
    agg = {
        'project': project, 'shard': os.path.basename(path), 'docs': 0, 'chars': 0,
        'ns_docs': collections.Counter(), 'ns_chars': collections.Counter(),
        'flag_docs': collections.Counter(), 'flag_chars': collections.Counter(),
        'flag_docs_main': collections.Counter(), 'flag_chars_main': collections.Counter(),
        'issue_chars': collections.Counter(), 'issue_chars_main': collections.Counter(),
        'issue_docs': collections.Counter(), 'issue_docs_main': collections.Counter(),
        'main_docs': 0, 'main_chars': 0, 'licenses': collections.Counter(),
        'meta_keys': collections.Counter(), 'examples': collections.defaultdict(list),
        'stop_ratio_hist_main': collections.Counter(), 'lengths': [], 'words': [],
        'lengths_main': [], 'words_main': [], 'ns_flag_docs': collections.Counter(),
        'created_year': collections.Counter(), 'created_year_main': collections.Counter(),
    }
    records = []
    minhash_pool = []
    seen_content = 0
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            doc = json.loads(line)
            meta = load_meta(doc)
            text = doc['text']
            ns = meta.get('namespace', '?')
            agg['docs'] += 1
            agg['chars'] += len(text)
            agg['ns_docs'][ns] += 1
            agg['ns_chars'][ns] += len(text)
            agg['licenses'][meta.get('license')] += 1
            year = str(doc.get('created', ''))[:4]
            agg['created_year'][year] += 1
            if ns == '0':
                agg['created_year_main'][year] += 1
            agg['meta_keys'].update(meta.keys())
            n_words = len(text.split())
            agg['lengths'].append(len(text))
            agg['words'].append(n_words)
            flags, chars, spans, body, body_words, stop_ratio, non_latin = analyze_doc(doc, meta)
            is_main = ns == '0'
            if is_main:
                agg['main_docs'] += 1
                agg['main_chars'] += len(text)
                agg['lengths_main'].append(len(text))
                agg['words_main'].append(n_words)
                if body_words >= 20:
                    agg['stop_ratio_hist_main'][round(stop_ratio, 2)] += 1
            for name in ISSUE_BITS:
                if flags & BIT[name]:
                    agg['flag_docs'][name] += 1
                    agg['flag_chars'][name] += len(text)
                    agg['ns_flag_docs'][(ns, name)] += 1
                    if is_main:
                        agg['flag_docs_main'][name] += 1
                        agg['flag_chars_main'][name] += len(text)
                    examples = agg['examples'][name]
                    if len(examples) < EXAMPLES_PER_ISSUE and rng.random() < 0.02:
                        start, stop = spans.get(name, (0, 0))
                        snippet_start = max(0, start - 100) if stop else 0
                        examples.append((doc['id'], ns, text[snippet_start:snippet_start + 280]))
            for key, value in chars.items():
                agg['issue_chars'][key] += value
                agg['issue_docs'][key] += 1
                if is_main:
                    agg['issue_chars_main'][key] += value
                    agg['issue_docs_main'][key] += 1
                if key.startswith(('markup:', 'holes:')) or key in ('empty_items', 'lang_label_list_lines', 'latex_math'):
                    examples = agg['examples'][key]
                    if len(examples) < EXAMPLES_PER_ISSUE and rng.random() < 0.05:
                        if key in spans:
                            start, stop = spans[key]
                        else:
                            start, stop = find_example_span(key, text)
                        examples.append((doc['id'], ns, text[max(0, start - 120):stop + 120]))
            records.append((int(doc['id'].split('-', 1)[1]), ns_int(ns), len(text), body_words, flags, digest(text),
                            digest(body) if len(body) >= 30 else 0))
            if is_main and body_words >= 50:
                seen_content += 1
                if len(minhash_pool) < MINHASH_PER_PROJECT:
                    minhash_pool.append((doc['id'], text))
                elif rng.random() < MINHASH_PER_PROJECT / seen_content:
                    minhash_pool[rng.randrange(MINHASH_PER_PROJECT)] = (doc['id'], text)
    np.save(os.path.join(OUT, 'records_' + os.path.basename(path) + '.npy'), np.array(records, dtype=RECORD_DTYPE))
    del records
    mults, adds = minhash_params()
    signed = []
    for doc_id, text in minhash_pool:
        hashes = shingle_hashes(text)
        if hashes is not None:
            signed.append((doc_id, text[:200], minhash_signature(hashes, mults, adds)))
    with open(os.path.join(OUT, 'minhash_' + os.path.basename(path) + '.pkl'), 'wb') as fh:
        pickle.dump(signed, fh, protocol=pickle.HIGHEST_PROTOCOL)
    agg['examples'] = dict(agg['examples'])
    return agg


def find_example_span(key, text):
    kind, _, name = key.partition(':')
    table = {'markup': dict(MARKUP_RES), 'holes': dict(HOLE_RES)}
    if kind in table:
        match = table[kind][name].search(text)
    elif key == 'empty_items':
        match = EMPTY_ITEM_RE.search(text)
    elif key == 'lang_label_list_lines':
        match = TRANSLATION_LINE_RE.search(text)
    else:
        match = LATEX_DISPLAY_RE.search(text)
    return match.span() if match else (0, 0)


def percentiles(values):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.int64)
    qs = [1, 5, 25, 50, 75, 95, 99]
    out = {f'p{q}': int(np.percentile(arr, q)) for q in qs}
    out['max'] = int(arr.max())
    out['mean'] = float(arr.mean())
    out['n'] = int(arr.size)
    out['sum'] = int(arr.sum())
    return out


def minhash_params():
    rng = np.random.default_rng(SEED)
    mults = rng.integers(1, 2 ** 63, size=MINHASH_PERMS, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    adds = rng.integers(0, 2 ** 63, size=MINHASH_PERMS, dtype=np.uint64)
    return mults, adds


def near_duplicates(pool_files):
    entries = []
    signatures = []
    for pool_path in pool_files:
        project = os.path.basename(pool_path).split('_', 2)[2].split('.')[0]
        with open(pool_path, 'rb') as fh:
            pool = pickle.load(fh)
        for doc_id, head, signature in pool:
            entries.append((project, os.path.basename(pool_path), doc_id, head))
            signatures.append(signature)
    sigs = np.vstack(signatures)
    buckets = collections.defaultdict(list)
    for index in range(len(sigs)):
        for band in range(LSH_BANDS):
            key = (band, sigs[index, band * LSH_ROWS:(band + 1) * LSH_ROWS].tobytes())
            buckets[key].append(index)
    pairs = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > 2000:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    confirmed = []
    for i, j in pairs:
        estimate = float(np.mean(sigs[i] == sigs[j]))
        if estimate >= NEAR_DUP_THRESHOLD:
            confirmed.append((i, j, estimate))
    parent = list(range(len(sigs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in confirmed:
        parent[find(i)] = find(j)
    clusters = collections.defaultdict(list)
    for index in range(len(sigs)):
        clusters[find(index)].append(index)
    per_project_docs = collections.Counter(entry[0] for entry in entries)
    in_cluster = collections.Counter()
    redundant = collections.Counter()
    for members in clusters.values():
        if len(members) < 2:
            continue
        for member in members:
            in_cluster[entries[member][0]] += 1
        for member in members[1:]:
            redundant[entries[member][0]] += 1
    examples = []
    random.Random(SEED).shuffle(confirmed)
    for i, j, estimate in confirmed[:25]:
        examples.append({'a': entries[i][2], 'a_project': entries[i][0], 'b': entries[j][2], 'b_project': entries[j][0],
                         'jaccard_est': estimate, 'a_head': entries[i][3][:160], 'b_head': entries[j][3][:160]})
    return {'docs': dict(per_project_docs), 'docs_in_near_dup_clusters': dict(in_cluster),
            'redundant_docs_beyond_first_in_cluster': dict(redundant), 'confirmed_pairs': len(confirmed),
            'cross_project_pairs': sum(1 for i, j, _ in confirmed if entries[i][0] != entries[j][0]),
            'method': f'MinHash {MINHASH_PERMS} perms (multiply-add mod 2^64 over xxh64 of lowercased word 5-shingles), '
                      f'LSH {LSH_BANDS}x{LSH_ROWS}, keep pairs with estimated Jaccard >= {NEAR_DUP_THRESHOLD}; '
                      f'sample = reservoir of up to {MINHASH_PER_PROJECT} main-namespace docs with >=50 body words per shard',
            'examples': examples}


def filtered_comparison(records_by_shard, filtered_files):
    """Match filtered docs to raw docs by (project, ns, page id); report drops and text changes."""
    # Each project maps to sorted record keys so lookups use np.searchsorted instead of a dict of millions of keys.
    by_project = collections.defaultdict(list)
    for shard, records in records_by_shard.items():
        by_project[project_of(shard)].append((shard, records))
    index = {}
    for project, items in by_project.items():
        keys = np.concatenate([page_key(r['page'], r['ns']) for _, r in items])
        digests = np.concatenate([r['digest'] for _, r in items])
        order = np.argsort(keys, kind='stable')
        index[project] = (keys[order], digests[order])
    matched = {project: {} for project in index}
    filtered_profile = []
    coverage = {}
    changed_samples = collections.defaultdict(list)
    rng = random.Random(SEED)
    for path in filtered_files:
        fname = os.path.basename(path)
        prof = {'file': fname, 'docs': 0, 'by_project': collections.Counter(), 'inversions': 0,
                'ns': collections.Counter(), 'meta_keys': collections.Counter(), 'licenses': collections.Counter(),
                'chars': 0, 'matched_to_raw_sample': 0}
        prev = {}
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            for line in fh:
                doc = json.loads(line)
                project = project_from_source(doc['source'])
                prof['docs'] += 1
                prof['chars'] += len(doc['text'])
                prof['by_project'][project] += 1
                ns, page = doc['id'].split('-', 1)
                page = int(page)
                prof['ns'][(project, ns)] += 1
                if page < prev.get(project, -1):
                    prof['inversions'] += 1
                prev[project] = page
                lo, hi = coverage.get((fname, project), (page, page))
                coverage[(fname, project)] = (min(lo, page), max(hi, page))
                if project not in index:
                    continue
                keys, digests = index[project]
                key = page * 10000 + ns_int(ns)
                pos = int(np.searchsorted(keys, key))
                if pos >= len(keys) or keys[pos] != key:
                    continue
                prof['matched_to_raw_sample'] += 1
                meta = load_meta(doc)
                prof['meta_keys'].update(meta.keys())
                prof['licenses'][meta.get('license')] += 1
                text = doc['text']
                same = digest(text) == int(digests[pos])
                matched[project][key] = (same, len(text))
                if not same and len(changed_samples[project]) < 300 and rng.random() < 0.5:
                    changed_samples[project].append((doc['id'], text))
        prof['by_project'] = dict(prof['by_project'])
        prof['ns'] = {f'{p}:{n}': c for (p, n), c in prof['ns'].most_common(40)}
        prof['meta_keys'] = dict(prof['meta_keys'])
        prof['licenses'] = dict(prof['licenses'])
        filtered_profile.append(prof)
        print('filtered file done', fname, prof['docs'], flush=True)

    # A raw doc inside a page-id range that a filtered file covers for that project, and absent from it, was dropped.
    ranges = collections.defaultdict(list)
    for (fname, project), span in coverage.items():
        ranges[project].append(span)
    results = {}
    for shard, records in records_by_shard.items():
        project = project_of(shard)
        in_range_mask = np.zeros(len(records), dtype=bool)
        for lo, hi in ranges.get(project, []):
            in_range_mask |= (records['page'] >= lo) & (records['page'] <= hi)
        in_range = records[in_range_mask]
        if len(in_range) == 0:
            results[shard] = {'raw_docs': int(len(records)), 'raw_docs_in_filtered_range': 0}
            continue
        keys = page_key(in_range['page'], in_range['ns'])
        hits = matched.get(project, {})
        kept_mask = np.array([int(k) in hits for k in keys], dtype=bool)
        same_mask = np.array([hits.get(int(k), (False, 0))[0] for k in keys], dtype=bool)
        filtered_len = np.array([hits.get(int(k), (False, 0))[1] for k in keys], dtype=np.int64)
        kept, dropped = in_range[kept_mask], in_range[~kept_mask]

        def crosstab(group):
            table = {'docs': int(len(group)), 'chars': int(group['chars'].sum()),
                     'ns': dict(collections.Counter(group['ns'].tolist()).most_common(20)),
                     'flags': {name: int(((group['flags'] & BIT[name]) != 0).sum()) for name in ISSUE_BITS}}
            return table

        def keep_rate(mask_values, labels):
            out = {}
            for label in labels:
                sel = mask_values == label
                out[str(label)] = [int((sel & kept_mask).sum()), int((sel & ~kept_mask).sum())]
            return out

        len_edges = np.array([0, 10, 20, 30, 50, 100, 200, 500, 10 ** 12])
        len_bucket = len_edges[np.searchsorted(len_edges, in_range['body_words'])]
        flag_rates = {}
        for name in ISSUE_BITS:
            sel = (in_range['flags'] & BIT[name]) != 0
            flag_rates[name] = [int((sel & kept_mask).sum()), int((sel & ~kept_mask).sum())]
        ns_labels = [ns for ns, _ in collections.Counter(in_range['ns'].tolist()).most_common(25)]
        dropped_ids = rng.sample(range(len(dropped)), min(60, len(dropped)))
        results[shard] = {
            'raw_docs': int(len(records)), 'raw_docs_in_filtered_range': int(len(in_range)),
            'kept': int(kept_mask.sum()), 'dropped': int((~kept_mask).sum()),
            'kept_unchanged_text': int((kept_mask & same_mask).sum()), 'kept_changed_text': int((kept_mask & ~same_mask).sum()),
            'raw_chars_in_range': int(in_range['chars'].sum()), 'kept_raw_chars': int(kept['chars'].sum()),
            'kept_filtered_chars': int(filtered_len[kept_mask].sum()),
            'kept_table': crosstab(kept), 'dropped_table': crosstab(dropped),
            'keep_rate_by_ns_[kept,dropped]': keep_rate(in_range['ns'], ns_labels),
            'keep_rate_by_body_words_le_[kept,dropped]': keep_rate(len_bucket, len_edges.tolist()),
            'keep_rate_by_flag_[kept,dropped]': flag_rates,
            'dropped_examples': [f"{int(dropped[i]['ns'])}-{int(dropped[i]['page'])}" for i in dropped_ids],
        }
    return results, filtered_profile, changed_samples, coverage


def diff_changed(changed_samples, raw_files):
    """Re-read raw texts for sampled changed docs and summarize the edit operations."""
    import difflib
    wanted = {(project, doc_id): text for project, items in changed_samples.items() for doc_id, text in items}
    summaries = collections.defaultdict(list)
    op_stats = collections.defaultdict(collections.Counter)
    for path in raw_files:
        project = project_of(path)
        if not any(key[0] == project for key in wanted):
            continue
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            for line in fh:
                doc = json.loads(line)
                key = (project, doc['id'])
                if key not in wanted:
                    continue
                raw, filt = doc['text'], wanted.pop(key)
                ops = []
                # Line-level alignment first; char-level alignment only inside changed line blocks keeps difflib fast.
                raw_lines, filt_lines = raw.split('\n'), filt.split('\n')
                line_matcher = difflib.SequenceMatcher(None, raw_lines, filt_lines, autojunk=False)
                for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
                    if tag == 'equal':
                        continue
                    raw_block, filt_block = '\n'.join(raw_lines[i1:i2]), '\n'.join(filt_lines[j1:j2])
                    if tag == 'replace' and len(raw_block) + len(filt_block) < 20000:
                        char_matcher = difflib.SequenceMatcher(None, raw_block, filt_block, autojunk=False)
                        pieces = [(t, raw_block[a1:a2], filt_block[b1:b2], raw_block[max(0, a1 - 40):a2 + 40])
                                  for t, a1, a2, b1, b2 in char_matcher.get_opcodes() if t != 'equal']
                    else:
                        pieces = [(tag, raw_block, filt_block, raw_block[:80])]
                    for piece_tag, raw_piece, filt_piece, context in pieces:
                        op_stats[project][piece_tag] += 1
                        op_stats[project]['raw_chars_touched'] += len(raw_piece)
                        op_stats[project]['filtered_chars_touched'] += len(filt_piece)
                        op_stats[project]['removed:' + raw_piece[:40]] += 1 if piece_tag == 'delete' else 0
                        op_stats[project]['inserted:' + filt_piece[:40]] += 1 if piece_tag in ('insert', 'replace') else 0
                        if len(ops) < 8:
                            ops.append((piece_tag, raw_piece[:150], filt_piece[:150], context[:200]))
                summaries[project].append({'id': doc['id'], 'raw_len': len(raw), 'filtered_len': len(filt),
                                           'ws_normalized_equal': normalize_for_hash(raw) == normalize_for_hash(filt),
                                           'ops': ops})
    stats = {}
    for project, counter in op_stats.items():
        totals = {k: v for k, v in counter.items() if ':' not in k}
        totals['top_removed'] = [(k[8:], v) for k, v in counter.most_common() if k.startswith('removed:') and v][:25]
        totals['top_inserted'] = [(k[9:], v) for k, v in counter.most_common() if k.startswith('inserted:') and v][:25]
        stats[project] = totals
    return {'samples': dict(summaries), 'op_stats': stats}


def exact_duplicates(records_by_shard, field):
    shards = list(records_by_shard)
    digests = np.concatenate([records_by_shard[s][field] for s in shards])
    chars = np.concatenate([records_by_shard[s]['chars'] for s in shards])
    shard_ids = np.concatenate([np.full(len(records_by_shard[s]), i, dtype=np.int32) for i, s in enumerate(shards)])
    pages = np.concatenate([records_by_shard[s]['page'] for s in shards])
    nss = np.concatenate([records_by_shard[s]['ns'] for s in shards])
    valid = digests != 0
    order = np.argsort(digests, kind='stable')
    order = order[valid[order]]
    sorted_digests = digests[order]
    is_dup = np.zeros(len(order), dtype=bool)
    is_dup[1:] = sorted_digests[1:] == sorted_digests[:-1]
    group_start = np.maximum.accumulate(np.where(~is_dup, np.arange(len(order)), 0))
    first_shard = shard_ids[order][group_start]
    dup_idx = order[is_dup]
    projects = np.array([project_of(s) for s in shards])
    out = {'valid_docs': int(valid.sum()), 'dup_docs_beyond_first': {}, 'dup_chars': {}, 'docs': {}, 'chars': {}}
    for i, shard in enumerate(shards):
        sel = shard_ids[dup_idx] == i
        out['dup_docs_beyond_first'][shard] = int(sel.sum())
        out['dup_chars'][shard] = int(chars[dup_idx][sel].sum())
        out['docs'][shard] = int(((shard_ids == i) & valid).sum())
        out['chars'][shard] = int(chars[(shard_ids == i) & valid].sum())
    cross = collections.Counter()
    for a, b in zip(projects[first_shard[is_dup]], projects[shard_ids[dup_idx]]):
        if a != b:
            cross[f'{a}->{b}'] += 1
    out['cross_project'] = dict(cross)
    group_sizes = collections.Counter(sorted_digests[is_dup].tolist())
    largest = []
    for value, count in group_sizes.most_common(12):
        pos = int(np.searchsorted(sorted_digests, np.uint64(value)))
        member = order[pos]
        largest.append((f'{shards[shard_ids[member]]}:{nss[member]}-{pages[member]}', count + 1))
    out['largest_groups'] = largest
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    raw_dir = os.path.join(WORKDIR, 'raw')
    filtered_dir = os.path.join(WORKDIR, 'filtered')
    raw_files = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.jsonl.gz'))
    filtered_files = sorted(os.path.join(filtered_dir, f) for f in os.listdir(filtered_dir) if f.endswith('.json.gz'))
    print('raw', [os.path.basename(f) for f in raw_files], 'filtered', [os.path.basename(f) for f in filtered_files], flush=True)
    # Large shards first so the two workers finish at about the same time.
    by_size = sorted(raw_files, key=os.path.getsize, reverse=True)
    with multiprocessing.Pool(2, maxtasksperchild=1) as pool:
        aggs = pool.map(analyze_shard, by_size, chunksize=1)
    print('shard pass done', time.strftime('%H:%M:%S'), flush=True)

    report = {'shards': {}, 'overall': {}}
    all_lengths, all_words = [], []
    for agg in aggs:
        all_lengths.extend(agg['lengths'])
        all_words.extend(agg['words'])
        shard_report = {k: v for k, v in agg.items() if k not in ('lengths', 'words', 'lengths_main', 'words_main')}
        shard_report['length_chars'] = percentiles(agg['lengths'])
        shard_report['length_words'] = percentiles(agg['words'])
        shard_report['length_chars_main'] = percentiles(agg['lengths_main'])
        shard_report['length_words_main'] = percentiles(agg['words_main'])
        shard_report['ns_flag_docs'] = {f'{ns}:{name}': c for (ns, name), c in agg['ns_flag_docs'].items()}
        for key in ('ns_docs', 'ns_chars', 'flag_docs', 'flag_chars', 'flag_docs_main', 'flag_chars_main', 'issue_chars',
                    'issue_chars_main', 'issue_docs', 'issue_docs_main', 'licenses', 'meta_keys', 'stop_ratio_hist_main',
                    'created_year', 'created_year_main'):
            shard_report[key] = dict(agg[key])
        report['shards'][agg['shard']] = shard_report
    report['overall']['length_chars'] = percentiles(all_lengths)
    report['overall']['length_words'] = percentiles(all_words)
    del all_lengths, all_words, aggs

    records_by_shard = {os.path.basename(p): np.load(os.path.join(OUT, 'records_' + os.path.basename(p) + '.npy'))
                        for p in raw_files}
    report['overall']['exact_dup_full_text'] = exact_duplicates(records_by_shard, 'digest')
    report['overall']['exact_dup_body'] = exact_duplicates(records_by_shard, 'body_digest')
    print('exact dup done', flush=True)

    pool_files = sorted(os.path.join(OUT, f) for f in os.listdir(OUT) if f.startswith('minhash_'))
    report['overall']['near_dup'] = near_duplicates(pool_files)
    print('near dup done', flush=True)

    if filtered_files:
        results, profile, changed_samples, coverage = filtered_comparison(records_by_shard, filtered_files)
        report['filtered'] = {'per_raw_shard': results, 'filtered_files': profile,
                              'coverage_page_id_ranges': {f'{k[0]}|{k[1]}': v for k, v in coverage.items()}}
        report['filtered']['changed_text_diffs'] = diff_changed(changed_samples, raw_files)
        print('filtered comparison done', flush=True)

    with open(os.path.join(OUT, 'report.json'), 'w') as fh:
        json.dump(report, fh, indent=1, default=str, ensure_ascii=False)
    print('wrote', os.path.join(OUT, 'report.json'))


# Supplementary measurements: run as `python3 -I analyze.py WORKDIR extra`.
WIKIVOYAGE_SECTIONS = ('Understand', 'Get in', 'Get around', 'See', 'Do', 'Buy', 'Eat', 'Drink', 'Sleep', 'Connect', 'Go next', 'Stay safe', 'Cope', 'Talk', 'Learn', 'Work')
EMPTY_VOYAGE_SECTION_RE = re.compile(r'(?m)^(%s)[ \t]*\n(?:[ \t]*\n)*(?=(?:%s)[ \t]*$|\Z)' % ('|'.join(WIKIVOYAGE_SECTIONS), '|'.join(WIKIVOYAGE_SECTIONS)))
EMPTY_LANG_SLOT_RE = re.compile(r"(?m)^ ?\* [A-Z][A-Za-z' ()-]{1,30}:[ \t,;]*$")
EMPTY_PHRASE_RE = re.compile(r'(?m)^ ?\* [^\n:]{1,80} : [.?!]? ?\( ?[?.!]? ?\)\s*$')
TEMPLATE_BRACES_RE = dict(MARKUP_RES)['template_braces']


def extra_shard(path):
    project = project_of(path)
    rng = random.Random(SEED + zlib.crc32(os.path.basename(path).encode()) + 1)
    out = collections.Counter()
    hole_lines = []
    seen_hole_lines = 0
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            doc = json.loads(line)
            meta = load_meta(doc)
            ns = meta.get('namespace')
            text = doc['text']
            body = split_body(text, meta.get('title', ''))
            out[f'docs_ns{ns}'] += 1
            out[f'chars_ns{ns}'] += len(text)
            if ns not in ('0', '104'):
                continue
            no_math = LATEX_DISPLAY_RE.sub(' ', text)
            braces = [m.span() for m in TEMPLATE_BRACES_RE.finditer(no_math)]
            if braces:
                out[f'ns{ns}_docs_template_braces_outside_math'] += 1
                out[f'ns{ns}_chars_template_braces_outside_math'] += span_union_length(braces)
            slots = [m.span() for m in EMPTY_LANG_SLOT_RE.finditer(body)]
            if slots:
                out[f'ns{ns}_docs_empty_label_slot'] += 1
                out[f'ns{ns}_lines_empty_label_slot'] += len(slots)
                out[f'ns{ns}_chars_empty_label_slot'] += span_union_length(slots)
            phrases = EMPTY_PHRASE_RE.findall(body)
            if phrases:
                out[f'ns{ns}_docs_empty_phrase_translation'] += 1
                out[f'ns{ns}_lines_empty_phrase_translation'] += len(phrases)
            if project == 'wikivoyage' and ns == '0':
                empties = EMPTY_VOYAGE_SECTION_RE.findall(body)
                present = sum(1 for sec in WIKIVOYAGE_SECTIONS if re.search(r'(?m)^%s[ \t]*$' % re.escape(sec), body))
                out['voyage_docs_with_std_section'] += 1 if present else 0
                out['voyage_std_sections_present'] += present
                if empties:
                    out['voyage_docs_with_empty_std_section'] += 1
                    out['voyage_empty_std_sections'] += len(empties)
                    for sec in empties:
                        out['voyage_empty_section:' + sec] += 1
            if ns == '104':
                out['page_docs'] += 1
                if body[:1].islower():
                    out['page_docs_start_lowercase'] += 1
                if body and body.rstrip()[-1:] not in '.!?"\'”’)]:;':
                    out['page_docs_end_unterminated'] += 1
                if not body:
                    out['page_docs_empty'] += 1
                first = body.split('\n', 1)[0]
                if 0 < len(first) <= 60 and re.search(r'\d', first) and first.upper() == first:
                    out['page_docs_first_line_caps_with_digit'] += 1
            if ns == '0' and project != 'wiktionary':
                for raw_line in body.split('\n'):
                    if any(regex.search(raw_line) for _, regex in HOLE_RES):
                        seen_hole_lines += 1
                        if len(hole_lines) < 40:
                            hole_lines.append((doc['id'], raw_line[:220]))
                        elif rng.random() < 40 / seen_hole_lines:
                            hole_lines[rng.randrange(40)] = (doc['id'], raw_line[:220])
    return os.path.basename(path), dict(out), hole_lines


def extra_main():
    raw_dir = os.path.join(WORKDIR, 'raw')
    raw_files = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.jsonl.gz'))
    by_size = sorted(raw_files, key=os.path.getsize, reverse=True)
    with multiprocessing.Pool(2, maxtasksperchild=1) as pool:
        results = pool.map(extra_shard, by_size, chunksize=1)
    report = {name: {'counts': counts, 'hole_line_sample': lines} for name, counts, lines in results}
    with open(os.path.join(OUT, 'report_extra.json'), 'w') as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print('wrote', os.path.join(OUT, 'report_extra.json'))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == 'extra':
        extra_main()
    else:
        main()

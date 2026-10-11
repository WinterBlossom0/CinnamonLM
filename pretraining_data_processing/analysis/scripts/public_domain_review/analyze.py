"""Full-corpus quality analysis of common-pile/public_domain_review (raw) against its _filtered version."""
import collections
import glob
import gzip
import hashlib
import json
import os
import re
import sys
import unicodedata

import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'common_pile', 'public_domain_review', 'v0') + os.sep
SCRATCH = os.path.dirname(os.path.abspath(__file__))
FILTERED_PATH = os.path.join(SCRATCH, 'filtered', 'public-domain-review-0000.json.gz')
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'common_pile', 'public_domain_review', '.cache')


def load_raw():
    incomplete = glob.glob(os.path.join(CACHE_DIR, '**', '*.incomplete'), recursive=True)
    if incomplete:
        sys.exit(f'incomplete download files present: {incomplete}')
    rows = []
    for path in sorted(glob.glob(RAW_DIR + '*.jsonl.gz')):
        with gzip.open(path, 'rt') as handle:
            for line in handle:
                record = json.loads(line)
                record['_shard'] = os.path.basename(path)
                rows.append(record)
    return rows


def load_filtered():
    with gzip.open(FILTERED_PATH, 'rt') as handle:
        return [json.loads(line) for line in handle]


def percentiles(values):
    marks = [1, 5, 25, 50, 75, 95, 99]
    array = np.array(values)
    parts = [f'p{mark}={int(np.percentile(array, mark, method="lower"))}' for mark in marks]
    return ', '.join(parts) + f', min={array.min()}, max={array.max()}, mean={array.mean():.0f}'


# Date line formats seen in the header: "Aug 9, 2011" (collections) and "April 17, 2024" (essays).
DATE_LINE = re.compile(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2}, \d{4}$')


def split_header(text):
    """Header is title, optional subtitle, optional byline, and the date line; it ends at the first date line in the first 6 lines."""
    lines = text.split('\n')
    for index, line in enumerate(lines[:6]):
        if DATE_LINE.match(line.strip()):
            header = '\n'.join(lines[:index + 1])
            return header, text[len(header):]
    return '', text


LICENSE_FOOTER = 'The text of this essay is published under a CC BY-SA license, see here for details.'

# Each issue regex is applied to the full raw text; char share counts matched spans (or matched whole lines for line patterns).
ISSUES = {
    'license_footer_line': re.compile(r'(?m)^The text of this essay is published under a CC BY-SA license, see here for details\.$'),
    'blank_line_runs_3plus': re.compile(r'\n{3,}'),
    # Footnote number glued to a word or closing punctuation; the notes themselves are absent from the text.
    'dangling_footnote_marker': re.compile(r'(?<=[a-z\.”"’\)\?!])(?<!\d\.)(?<![A-Z]\.)(?<!\bNo\.)(?<!\bno\.)(?<!\bp\.)(?<!\bpp\.)(?<!\bvol\.)\d{1,3}(?=\s|$)'),
    # Line breaks lost when HTML <br> was stripped: verse or list lines run together with no space.
    'joined_lines_punct': re.compile(r'[a-z][.!?][A-Z][a-z]'),
    'joined_lines_camel': re.compile(r'\b[a-z]{3,}[A-Z][a-z]{2,}'),
    'joined_lines_any': re.compile(r'[a-z][.!?,;:][A-Z][a-z]|\b[a-z]{3,}[A-Z][a-z]{2,}'),
    'webpage_instruction': re.compile(r'(?i)\bscroll (?:through|down)\b|\bclick (?:on|here|the)\b|\bembedded (?:reader|player|viewer)\b|\bzoom in\b'),
    'raw_html_tag': re.compile(r'</?(?:a|p|div|span|img|br|em|strong|iframe|sup|blockquote|li|ul)\b[^>\n]*>'),
    'markdown_link': re.compile(r'\[[^\]\n]+\]\(https?://[^)\s]*[^)]*\)'),
    'bare_url': re.compile(r'https?://\S+|www\.\S+'),
    'email_or_masked_email': re.compile(r'[\w.*+-]+\s?@[\w*-]+\.[\w.]+'),
    'email_header_block': re.compile(r'Forwarded message|(?:From|To|Cc|Bcc|Subject):\S'),
    'wikipedia_copied_paragraph': re.compile(r'(?m)^.*\(Wikipedia\)\s*$'),
    'separator_line': re.compile(r'(?m)^[ \t]*(?:\*+|-{3,}|\* \* \*|_{3,})[ \t]*$'),
    'media_nav_line': re.compile(r'(?im)^.*\b(?:MP3 Download|Internet Archive Link|Download|Animated GIFs Collection)\b.{0,80}$'),
    # Whole line is a short call to action that refers to a link, the shop, comments, or an image gallery.
    'cta_line_short': re.compile(
        r'(?im)^(?=.{0,220}$).*(?:\b(?:see|click|read|find|view|watch|listen|browse|available|download)\b[^.\n]{0,80}\bhere\b'
        r'|\bour (?:online )?shop\b|\bin the comments\b|\blet us know\b|\bGIFs? Collection\b|\bpublished here under a CC\b).*$'),
    'cta_phrase_any': re.compile(
        r'(?i)\b(?:see|click|read|find|view|watch|listen|browse|available|download)\b[^.\n]{0,80}\bhere\b'
        r'|\bour (?:online )?shop\b|\bin the comments\b|\blet us know\b'),
    'refers_to_missing_media': re.compile(r'(?i)\b(?:featured|shown|seen|pictured|gathered|collected|browse|embedded|gallery|selection|images?|photographs?|film|video|recording)\b[^.\n]{0,60}\b(?:above|below)\b'),
    'credit_or_source_line': re.compile(r'(?im)^(?=.{0,250}$).*(?:\bSource\s*:|\bARC Identifier\b|\bLocal Identifier\b|\bImage credit\b|\bPhoto(?:graph)?\s*:).*$'),
    'okkult_boilerplate': re.compile(r'(?m)^.*Okkult Motion Pictures.*$'),
    'nbsp_or_zero_width': re.compile('[ ​‌‍⁠﻿­]'),
    'control_chars': re.compile('[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]'),
    'mojibake': re.compile('â€|Ã[\u0080-¿]|Â[ -¿]|�'),
}

ENGLISH_STOPWORDS = set('''the of and to a in is that it was for on as with by he his at from this be which are or an had not but have they were their its her she we you one all has been there who would so if more what when can no will about into some them out only other my these than up then also our your him those such any most over'''.split())
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")


def language_features(text):
    letters = [char for char in text if char.isalpha()]
    ascii_letters = sum(1 for char in letters if char.isascii())
    words = [word.lower() for word in WORD.findall(text)]
    stop_hits = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    ascii_ratio = ascii_letters / len(letters) if letters else 0.0
    stop_ratio = stop_hits / len(words) if words else 0.0
    return ascii_ratio, stop_ratio, len(words)


def normalize_for_hash(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', text).strip()


def shingles(text, size=5):
    words = re.findall(r'\w+', normalize_for_hash(text))
    if len(words) < size:
        return {' '.join(words)} if words else set()
    return {' '.join(words[i:i + size]) for i in range(len(words) - size + 1)}


NUM_PERM = 128


def minhash_signatures(shingle_sets):
    generator = np.random.default_rng(12345)
    coef_a = generator.integers(1, np.iinfo(np.int64).max, NUM_PERM, dtype=np.uint64) | np.uint64(1)
    coef_b = generator.integers(0, np.iinfo(np.int64).max, NUM_PERM, dtype=np.uint64)
    signatures = np.full((len(shingle_sets), NUM_PERM), np.iinfo(np.uint64).max, dtype=np.uint64)
    for row, shingle_set in enumerate(shingle_sets):
        if not shingle_set:
            continue
        hashes = np.array([int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little') for s in shingle_set],
                          dtype=np.uint64)
        # Multiply-add modulo 2**64 (uint64 wraparound) with odd multipliers is a universal hash family; the wrap is intended.
        with np.errstate(over='ignore'):
            values = hashes[:, None] * coef_a[None, :] + coef_b[None, :]
        signatures[row] = values.min(axis=0)
    return signatures


def jaccard(first, second):
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)


def report_issue(name, regex, rows, total_chars, out):
    docs_hit = 0
    chars_hit = 0
    matches_total = 0
    examples = []
    for record in rows:
        text = record['text']
        found = list(regex.finditer(text))
        if not found:
            continue
        docs_hit += 1
        matches_total += len(found)
        chars_hit += sum(match.end() - match.start() for match in found)
        if len(examples) < 4:
            match = found[0]
            snippet = text[max(0, match.start() - 80):match.end() + 60]
            examples.append(f'[{record["id"]}] {snippet!r}'[:290])
    out['issues'][name] = {
        'docs': docs_hit, 'doc_frac': docs_hit / len(rows), 'matches': matches_total,
        'chars': chars_hit, 'char_frac': chars_hit / total_chars, 'examples': examples,
    }


def repeated_paragraphs(rows, min_len=40, min_docs=3):
    doc_count = collections.Counter()
    for record in rows:
        for paragraph in {line.strip() for line in record['text'].split('\n') if len(line.strip()) >= min_len}:
            doc_count[paragraph] += 1
    repeated = {paragraph for paragraph, count in doc_count.items() if count >= min_docs}
    docs_hit = 0
    chars_hit = 0
    for record in rows:
        lines = [line.strip() for line in record['text'].split('\n')]
        hit = [line for line in lines if line in repeated]
        if hit:
            docs_hit += 1
            chars_hit += sum(len(line) + 1 for line in hit)
    top = [(count, paragraph[:150]) for paragraph, count in doc_count.most_common(15) if count >= min_docs]
    return docs_hit, chars_hit, top


def main():
    rows = load_raw()
    out = {'issues': {}}
    total_chars = sum(len(record['text']) for record in rows)
    out['doc_count'] = len(rows)
    out['per_shard'] = dict(collections.Counter(record['_shard'] for record in rows))
    out['per_type'] = dict(collections.Counter(record.get('type') for record in rows))
    out['total_chars'] = total_chars
    char_lengths = [len(record['text']) for record in rows]
    word_lengths = [len(record['text'].split()) for record in rows]
    out['chars'] = percentiles(char_lengths)
    out['words'] = percentiles(word_lengths)
    out['total_words'] = sum(word_lengths)
    for doc_type in sorted(out['per_type']):
        subset = [len(record['text']) for record in rows if record.get('type') == doc_type]
        out[f'chars_{doc_type}'] = percentiles(subset) + f', total={sum(subset)}'

    out['schema_fields'] = sorted({key for record in rows for key in record if not key.startswith('_')})
    out['metadata_fields'] = sorted({key for record in rows for key in record['metadata']})
    out['licenses'] = dict(collections.Counter(record['metadata'].get('license') for record in rows))
    out['url_prefixes'] = dict(collections.Counter(re.sub(r'^(https://publicdomainreview\.org/[^/]+/).*', r'\1', record['metadata']['url'])
                                                   for record in rows))
    out['missing_author'] = sum(1 for record in rows if not record.get('author'))

    # Header and body.
    header_found = 0
    header_chars = 0
    body_lengths = {}
    for record in rows:
        header, body = split_header(record['text'])
        if header:
            header_found += 1
            header_chars += len(header)
        body_lengths[record['id']] = len(body.strip())
    out['header_found_docs'] = header_found
    out['header_chars'] = header_chars
    out['header_char_frac'] = header_chars / total_chars
    for limit in (1, 100, 200, 300, 500, 1000):
        stubs = [doc_id for doc_id, length in body_lengths.items() if length < limit]
        stub_chars = sum(len(record['text']) for record in rows if record['id'] in set(stubs))
        out[f'body_lt_{limit}'] = {'docs': len(stubs), 'doc_frac': len(stubs) / len(rows), 'chars': stub_chars,
                                   'char_frac': stub_chars / total_chars, 'ids': stubs[:8]}

    for name, regex in ISSUES.items():
        report_issue(name, regex, rows, total_chars, out)

    import random
    sampler = random.Random(7)
    for name in ('dangling_footnote_marker', 'joined_lines_any'):
        all_matches = [(record['id'], record['text'][max(0, m.start() - 40):m.end() + 15])
                       for record in rows for m in ISSUES[name].finditer(record['text'])]
        out['issues'][name]['precision_review_sample'] = [f'[{doc_id}] {snippet!r}' for doc_id, snippet in sampler.sample(all_matches, min(40, len(all_matches)))]
        out['issues'][name]['by_type'] = dict(collections.Counter(next(r.get('type') for r in rows if r['id'] == doc_id) for doc_id, _ in all_matches))

    para_docs, para_chars, top_paras = repeated_paragraphs(rows)
    out['issues']['repeated_line_in_3plus_docs'] = {'docs': para_docs, 'doc_frac': para_docs / len(rows), 'chars': para_chars,
                                                    'char_frac': para_chars / total_chars, 'examples': top_paras}

    # Line structure: share of non-empty lines shorter than 60 chars, to find list/catalogue-like documents.
    listy = []
    for record in rows:
        _, body = split_header(record['text'])
        lines = [line for line in body.split('\n') if line.strip()]
        if len(lines) >= 10:
            short_share = sum(1 for line in lines if len(line.strip()) < 60) / len(lines)
            if short_share > 0.5:
                listy.append((record['id'], round(short_share, 2), len(record['text'])))
    out['list_like_docs_short_line_share_gt_0.5'] = {'docs': len(listy), 'chars': sum(item[2] for item in listy),
                                                     'char_frac': sum(item[2] for item in listy) / total_chars, 'examples': listy[:12]}

    # Language heuristics.
    language = {}
    for record in rows:
        language[record['id']] = language_features(record['text'])
    ascii_ratios = [value[0] for value in language.values()]
    stop_ratios = [value[1] for value in language.values() if value[2] >= 20]
    out['ascii_letter_ratio'] = percentiles([int(value * 1000) for value in ascii_ratios]) + ' (per mille)'
    out['stopword_ratio_docs_ge20words'] = percentiles([int(value * 1000) for value in stop_ratios]) + ' (per mille)'
    low_stop = sorted([(doc_id, round(value[1], 3), value[2]) for doc_id, value in language.items() if value[2] >= 20 and value[1] < 0.25],
                      key=lambda item: item[1])
    low_ascii = sorted([(doc_id, round(value[0], 3)) for doc_id, value in language.items() if value[0] < 0.97], key=lambda item: item[1])
    out['docs_stopword_ratio_lt_0.25'] = low_stop
    out['docs_ascii_letter_ratio_lt_0.97'] = low_ascii
    # Line-level: non-empty lines of >= 8 words whose English stopword ratio is < 0.1 (foreign-language passages inside English docs).
    foreign_lines = 0
    foreign_chars = 0
    foreign_docs = set()
    foreign_examples = []
    for record in rows:
        for line in record['text'].split('\n'):
            words = [word.lower() for word in WORD.findall(line)]
            if len(words) < 8:
                continue
            ratio = sum(1 for word in words if word in ENGLISH_STOPWORDS) / len(words)
            if ratio < 0.1:
                foreign_lines += 1
                foreign_chars += len(line)
                foreign_docs.add(record['id'])
                if len(foreign_examples) < 10:
                    foreign_examples.append(f'[{record["id"]}] {line[:160]!r}')
    out['low_stopword_lines'] = {'lines': foreign_lines, 'docs': len(foreign_docs), 'chars': foreign_chars,
                                 'char_frac': foreign_chars / total_chars, 'examples': foreign_examples}

    # Exact duplicates on normalized full text and on normalized body (header removed).
    full_hash = collections.defaultdict(list)
    body_hash = collections.defaultdict(list)
    for record in rows:
        full_hash[hashlib.sha1(normalize_for_hash(record['text']).encode()).hexdigest()].append(record['id'])
        body = split_header(record['text'])[1]
        if body.strip():
            body_hash[hashlib.sha1(normalize_for_hash(body).encode()).hexdigest()].append(record['id'])
    out['exact_dup_groups_fulltext'] = [ids for ids in full_hash.values() if len(ids) > 1]
    out['exact_dup_groups_body'] = [ids for ids in body_hash.values() if len(ids) > 1]
    out['empty_body_docs'] = sum(1 for length in body_lengths.values() if length == 0)

    # Near duplicates: MinHash (128 perms) on word 5-shingles of the body, then exact Jaccard on candidate pairs.
    ids = [record['id'] for record in rows]
    shingle_sets = [shingles(split_header(record['text'])[1]) for record in rows]
    signatures = minhash_signatures(shingle_sets)
    candidates = []
    for start in range(len(rows)):
        agreement = (signatures[start + 1:] == signatures[start]).mean(axis=1)
        for offset in np.nonzero(agreement >= 0.3)[0]:
            other = start + 1 + int(offset)
            if shingle_sets[start] and shingle_sets[other]:
                candidates.append((start, other, float(agreement[offset])))
    near_pairs = []
    for first, second, estimate in candidates:
        exact = jaccard(shingle_sets[first], shingle_sets[second])
        containment = len(shingle_sets[first] & shingle_sets[second]) / min(len(shingle_sets[first]), len(shingle_sets[second]))
        near_pairs.append((ids[first], ids[second], round(estimate, 3), round(exact, 3), round(containment, 3)))
    out['near_dup_pairs_minhash_est_ge_0.3'] = near_pairs
    out['near_dup_pairs_jaccard_ge_0.5'] = [pair for pair in near_pairs if pair[3] >= 0.5]
    out['near_dup_pairs_jaccard_ge_0.8'] = [pair for pair in near_pairs if pair[3] >= 0.8]

    # Raw vs filtered.
    filtered = load_filtered()
    raw_by_id = {record['id']: record for record in rows}
    filtered_by_id = {record['id']: record for record in filtered}
    dropped = [doc_id for doc_id in raw_by_id if doc_id not in filtered_by_id]
    added = [doc_id for doc_id in filtered_by_id if doc_id not in raw_by_id]
    identical = 0
    edit_kinds = collections.Counter()
    edit_examples = []
    removed_chars = 0
    for doc_id, record in filtered_by_id.items():
        if doc_id not in raw_by_id:
            continue
        raw_text = raw_by_id[doc_id]['text']
        new_text = record['text']
        if raw_text == new_text:
            identical += 1
            continue
        prefix = len(os.path.commonprefix([raw_text, new_text]))
        suffix = len(os.path.commonprefix([raw_text[prefix:][::-1], new_text[prefix:][::-1]]))
        raw_span = raw_text[prefix:len(raw_text) - suffix]
        new_span = new_text[prefix:len(new_text) - suffix]
        removed_chars += len(raw_text) - len(new_text)
        has_footer = LICENSE_FOOTER in raw_text and LICENSE_FOOTER not in new_text
        has_email = '<EMAIL_ADDRESS>' in new_text and '<EMAIL_ADDRESS>' not in raw_text
        edit_kinds[('license_footer_removed' if has_footer else '') + ('+email_redacted' if has_email else '')] += 1
        if has_email or not has_footer:
            edit_examples.append(f'[{doc_id}] {raw_span[:200]!r} -> {new_span[:200]!r}')
        stripped_check = raw_text.replace(LICENSE_FOOTER, '')
        if has_footer and not has_email and stripped_check != new_text:
            edit_kinds['footer_removal_not_exact_substring_delete'] += 1
    other_field_changes = collections.Counter()
    for doc_id, record in filtered_by_id.items():
        raw_record = raw_by_id.get(doc_id)
        if raw_record is None:
            continue
        for key in ('added', 'author', 'date', 'source', 'type'):
            if raw_record.get(key) != record.get(key):
                other_field_changes[key] += 1
        for key in ('license', 'url'):
            if raw_record['metadata'].get(key) != record['metadata'].get(key):
                other_field_changes['metadata.' + key] += 1
    out['filtered'] = {
        'filtered_docs': len(filtered), 'dropped': [(doc_id, len(raw_by_id[doc_id]['text']), body_lengths[doc_id]) for doc_id in dropped],
        'dropped_chars': sum(len(raw_by_id[doc_id]['text']) for doc_id in dropped), 'added': added,
        'identical_text': identical, 'modified_text': len(filtered) - identical - len(added), 'edit_kinds': dict(edit_kinds),
        'edit_examples': edit_examples, 'chars_removed_from_kept_docs': removed_chars,
        'filtered_total_chars': sum(len(record['text']) for record in filtered),
        'filtered_metadata_fields': sorted({key for record in filtered for key in record['metadata']}),
        'other_field_changes': dict(other_field_changes),
        'raw_stubs_body_lt_200_kept': [doc_id for doc_id, length in body_lengths.items() if length < 200 and doc_id in filtered_by_id],
    }
    # Issues still present after filtering.
    remaining = {}
    filtered_total = out['filtered']['filtered_total_chars']
    for name, regex in ISSUES.items():
        docs_hit = 0
        chars_hit = 0
        for record in filtered:
            found = list(regex.finditer(record['text']))
            if found:
                docs_hit += 1
                chars_hit += sum(match.end() - match.start() for match in found)
        remaining[name] = {'docs': docs_hit, 'doc_frac': round(docs_hit / len(filtered), 4), 'char_frac': round(chars_hit / filtered_total, 5)}
    out['filtered']['issues_remaining'] = remaining
    out['filtered']['redaction_token_docs'] = [(record['id'], record['text'].count('<EMAIL_ADDRESS>')) for record in filtered if '<EMAIL_ADDRESS>' in record['text']]
    out['filtered']['masked_email_left_after_redaction'] = [record['id'] for record in filtered if re.search(r'\*{3,}\s?@', record['text'])]

    json.dump(out, sys.stdout, indent=1, ensure_ascii=False, default=str)


if __name__ == '__main__':
    main()

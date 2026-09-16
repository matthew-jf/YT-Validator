"""Predict WESS language_id for unprocessed claims.

Precision-first cascade, each tier only fires above a cutoff tuned on a
temporal holdout of labeled history (same triage philosophy as pipeline.py):

  1. CHANNEL  - the channel's historical claims all carry one language_id
  2. TITLE    - the title contains a validated language-name rule
                (Anglicized names from sheets_language_families.csv plus
                native-name aliases mined from history, e.g. "bahasa melayu jambi");
                when several languages are named, the longest name wins, and
                the last-named breaks a tie
  3. FASTTEXT - supervised fastText classifier trained on historical
                (video_title -> language_id) pairs
  4. LID      - pretrained lid.176 language ID on the title, ISO -> WESS via
                the sheets mapping, ambiguous ISO resolved by history frequency
  5. REVIEW   - no tier confident enough; route to manual review

Training requires --history (an all_claims export with language_id). The fitted
signals and tuned cutoffs are cached to wess_artifact.json + wess_fasttext.ftz;
delete both to retrain. Rows whose video_id appears in --eval-labels are
excluded from training so the evaluation stays honest.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas

BASE_DIR = Path(__file__).resolve().parent
SHEETS_PATH = BASE_DIR / 'sheets_language_families.csv'
ARTIFACT_PATH = BASE_DIR / 'wess_artifact.json'
FT_MODEL_PATH = BASE_DIR / 'wess_fasttext.ftz'
LID_MODEL_PATH = BASE_DIR / 'lid.176.ftz'
LID_URL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'

PRECISION_TARGET = 0.95   # per-tier precision required on the tuning set
MIN_BUCKET_N = 25         # a tier must fire at least this often on the tuning set
VAL_DAYS = 120            # temporal-holdout fallback when no reviewed batch is given
CASCADE = ['CHANNEL', 'TITLE', 'ASR', 'FASTTEXT', 'LID']
MAX_TITLE_WORDS = 3       # longest phrase matched in a title, and indexed from a name
ASR_MIN_PER_LANG = 25     # tuning rows an ASR language needs before it is judged
                          # (matches MIN_BUCKET_N: 5/5 passes on luck alone)
ASR_PRECISION = 0.95      # ...and the precision it must reach to be trusted - the same bar as
                          # every other tier: ASR runs before FASTTEXT (~92% held out) and may
                          # override TITLE, so a lower bar would trade accuracy for coverage
ASR_DEFAULT_LIMIT = 100   # captions.list lookups per PHASE - tuning and prediction each get
                          # this many. 50 quota units per lookup, so a full run costs at most
                          # 2 x 100 x 50 = 10,000 units: exactly the daily key that production
                          # shares. Uncapped, tuning alone would ask for 8 days of quota.
ASR_DAILY_LIMIT = 180     # lookups per day for the collector: 9,000 of the 10,000 units, leaving
                          # room for the verdict pipeline's availability checks (~48 per run)
ASR_CACHE_PATH = BASE_DIR / 'data' / 'asr_cache.jsonl'
ASR_STATUS_PATH = BASE_DIR / 'data' / 'asr_collector_status.json'
# fastText is reproducible only with a fixed seed AND one thread: its threads update the
# model without locking, so with 7 threads 1 prediction in 10 changed between identical
# runs (seed set). One thread makes retrains identical; a full retrain takes ~12 min, not ~5.
FT_SEED = 1
FT_THREADS = 1
TRIAGE_PRIORITY = {'AUTO_Y': 0, 'REVIEW': 1}   # then rows with no triage, then the near-certain N
TRUTHY = {'true', '1', 'yes', 't', 'y'}

# fastText lid.176 labels are mostly ISO 639-1; the sheets mapping uses 639-3.
# Candidates are tried in order until one exists in the mapping.
ISO1_TO_3 = {
    'af': ['afr'], 'am': ['amh'], 'ar': ['arb', 'ara'], 'as': ['asm'],
    'az': ['azj', 'aze'], 'ba': ['bak'], 'be': ['bel'], 'bg': ['bul'],
    'bn': ['ben'], 'bo': ['bod'], 'br': ['bre'], 'bs': ['bos'], 'ca': ['cat'],
    'ce': ['che'], 'co': ['cos'], 'cs': ['ces'], 'cv': ['chv'], 'cy': ['cym'],
    'da': ['dan'], 'de': ['deu'], 'dv': ['div'], 'el': ['ell'], 'en': ['eng'],
    'eo': ['epo'], 'es': ['spa'], 'et': ['ekk', 'est'], 'eu': ['eus'],
    'fa': ['pes', 'fas'], 'fi': ['fin'], 'fr': ['fra'], 'fy': ['fry'],
    'ga': ['gle'], 'gd': ['gla'], 'gl': ['glg'], 'gn': ['gug', 'grn'],
    'gu': ['guj'], 'gv': ['glv'], 'he': ['heb'], 'hi': ['hin'], 'hr': ['hrv'],
    'ht': ['hat'], 'hu': ['hun'], 'hy': ['hye'], 'id': ['ind'], 'is': ['isl'],
    'it': ['ita'], 'ja': ['jpn'], 'jv': ['jav'], 'ka': ['kat'], 'kk': ['kaz'],
    'km': ['khm'], 'kn': ['kan'], 'ko': ['kor'], 'ku': ['kmr', 'kur'],
    'kv': ['kom'], 'kw': ['cor'], 'ky': ['kir'], 'la': ['lat'], 'lb': ['ltz'],
    'li': ['lim'], 'lo': ['lao'], 'lt': ['lit'], 'lv': ['lvs', 'lav'],
    'mg': ['plt', 'mlg'], 'mk': ['mkd'], 'ml': ['mal'], 'mn': ['khk', 'mon'],
    'mr': ['mar'], 'ms': ['zlm', 'zsm', 'msa'], 'mt': ['mlt'], 'my': ['mya'],
    'ne': ['npi', 'nep'], 'nl': ['nld'], 'nn': ['nno'], 'no': ['nob', 'nor'],
    'oc': ['oci'], 'or': ['ory'], 'os': ['oss'], 'pa': ['pan'], 'pl': ['pol'],
    'ps': ['pbt', 'pbu', 'pus'], 'pt': ['por'], 'qu': ['quy', 'que'],
    'rm': ['roh'], 'ro': ['ron'], 'ru': ['rus'], 'sa': ['san'], 'sc': ['srd'],
    'sd': ['snd'], 'sh': ['hbs', 'srp'], 'si': ['sin'], 'sk': ['slk'],
    'sl': ['slv'], 'so': ['som'], 'sq': ['als', 'sqi'], 'sr': ['srp'],
    'su': ['sun'], 'sv': ['swe'], 'sw': ['swh', 'swa'], 'ta': ['tam'],
    'te': ['tel'], 'tg': ['tgk'], 'th': ['tha'], 'tk': ['tuk'], 'tl': ['tgl'],
    'tr': ['tur'], 'tt': ['tat'], 'ug': ['uig'], 'uk': ['ukr'], 'ur': ['urd'],
    'uz': ['uzn', 'uzb'], 'vi': ['vie'], 'wa': ['wln'], 'yi': ['ydd', 'yid'],
    'yo': ['yor'], 'zh': ['cmn', 'zho'], 'yue': ['yue'],
}


# ---------------------------------------------------------------------------
# Normalization / loading
# ---------------------------------------------------------------------------
def normalize_id(value):
    """Strip the apostrophe Excel adds to ids that begin with '-'."""
    return str(value).strip().lstrip("'")


def norm_lang(value):
    """'6464', 6464.0 -> '6464'; empty / 0 / NaN -> None."""
    if pandas.isna(value):
        return None
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        return None
    return str(num) if num > 0 else None


def norm_title(title):
    """Whitespace-safe fastText input: lowercase, punctuation split off words."""
    text = str(title).lower()
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    return ' '.join(text.split())


def canon_phrase(text):
    """Lowercased words of a phrase, sorted, so word order stops mattering.

    A language name is a set of words, not a sequence: the sheet says "Haitian
    Creole" and a title may say "Creole Haitian". Both canonicalise to
    "creole haitian" and match the same rule.
    """
    return ' '.join(sorted(re.findall(r'\w+', str(text).lower())))


def title_phrases(title, max_words=MAX_TITLE_WORDS):
    """All 1..max_words word n-grams of the title, canonicalised, longest first."""
    tokens = re.findall(r'\w+', str(title).lower())
    for n in range(min(max_words, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            yield ' '.join(sorted(tokens[i:i + n]))


# Every column of the sheet that carries a name a person might type in a title.
# Anglicized_name is the display name; the others are the production name
# ("LESSER ANTILLEAN CREOLE FRENCH"), the World Christian Database name, and the
# dialect. Titles use all of them, so all of them are matchable.
NAME_COLUMNS = ('Anglicized_name', 'Language_JFProd', 'Language_name_WCD', 'Dialect_name')


def load_mapping():
    """sheets_language_families.csv -> (wess2name, name2wess, iso2wess).

    name2wess indexes each name AND its 2..MAX_TITLE_WORDS word windows, every
    key canonicalised by canon_phrase(), because titles reorder and abbreviate:
    "Creole French Lesser Antillean" never equals "Lesser Antillean Creole
    French" as a string, but names the same language and shares its words.
    """
    sheet = pandas.read_csv(SHEETS_PATH, dtype=str).fillna('')
    wess2name, name2wess, iso2wess = {}, defaultdict(set), defaultdict(set)
    for _, row in sheet.iterrows():
        wess = norm_lang(row['WESS_LAN_num'])
        if not wess:
            continue
        display = row.get('Anglicized_name', '').strip()
        wess2name.setdefault(wess, display)
        for column in NAME_COLUMNS:
            name = str(row.get(column, '')).strip().lower()
            if len(name) < 3:  # blank or near-blank names would match everything
                continue
            name2wess[canon_phrase(name)].add(wess)
            tokens = re.findall(r'\w+', name)
            for n in range(2, min(MAX_TITLE_WORDS, len(tokens)) + 1):
                for i in range(len(tokens) - n + 1):
                    name2wess[' '.join(sorted(tokens[i:i + n]))].add(wess)
        iso = row['ISO_lang'].strip().lower()
        if iso:
            iso2wess[iso].add(wess)
    return wess2name, dict(name2wess), dict(iso2wess)


def load_history(path, exclude_video_ids=()):
    """Labeled claims from an all_claims export -> DataFrame(channel_id, title, lang, created).

    Uses the stdlib csv reader: these exports contain rows that pandas' C
    parser rejects as malformed.
    """
    import csv
    import sys
    csv.field_size_limit(sys.maxsize)
    excluded = set(exclude_video_ids)
    rows = []
    with open(path, newline='', encoding='utf-8', errors='replace') as handle:
        for row in csv.DictReader(handle):
            lang = norm_lang(row.get('language_id') or None)
            if lang is None or row.get('video_id') in excluded:
                continue
            rows.append((row.get('channel_id') or '', row.get('video_title') or '',
                         lang, row.get('claim_created_date') or ''))
    df = pandas.DataFrame(rows, columns=['channel_id', 'video_title', 'lang', 'created'])
    df['created'] = pandas.to_datetime(df['created'], format='mixed', errors='coerce')
    return df


# ---------------------------------------------------------------------------
# Tier 1: channel unanimity
# ---------------------------------------------------------------------------
def build_channel_map(df, min_count):
    """channel_id -> language_id for channels whose labeled history is unanimous."""
    out = {}
    for channel, langs in df.groupby('channel_id')['lang']:
        counts = Counter(langs)
        if len(counts) == 1:
            lang, n = next(iter(counts.items()))
            if n >= min_count:
                out[channel] = lang
    return out


# ---------------------------------------------------------------------------
# Tier 2: title language-name rules
# ---------------------------------------------------------------------------
def title_match_stats(df, phrases):
    """phrase -> Counter(true language) over titles containing the phrase."""
    stats = defaultdict(Counter)
    for title, lang in zip(df['video_title'], df['lang']):
        seen = set()
        for phrase in title_phrases(title):
            if phrase in phrases and phrase not in seen:
                stats[phrase][lang] += 1
                seen.add(phrase)
    return stats


def mine_bahasa_aliases(df):
    """Candidate native-name phrases: 1-3 words following 'bahasa' in titles."""
    pattern = re.compile(r'bahasa((?:\s+\w+){1,3})')
    phrases = set()
    for title in df['video_title']:
        match = pattern.search(str(title).lower())
        if match:
            words = match.group(1).split()
            for n in range(1, len(words) + 1):
                phrases.add(canon_phrase('bahasa ' + ' '.join(words[:n])))
    return phrases


def build_title_rules(name2wess, wess_freq, train_df, cfg):
    """phrase -> (language_id, confidence). Only rules the history supports.

    Rules come from two sources: Anglicized names in the sheets mapping and
    mined 'bahasa X' native-name phrases. A rule is kept when history shows it
    precise (>= min_n matches at >= min_prec precision, retargeted to the
    majority label), or - for mapping names never seen in history - when the
    name is long and unambiguous enough to trust on its own.
    """
    candidates = {n for n in name2wess if len(n) >= 3}
    candidates |= mine_bahasa_aliases(train_df)
    stats = title_match_stats(train_df, candidates)

    rules = {}
    for phrase in candidates:
        counts = stats.get(phrase)
        if counts and sum(counts.values()) >= cfg['min_n']:
            lang, hits = counts.most_common(1)[0]
            total = sum(counts.values())
            if hits / total >= cfg['min_prec']:
                rules[phrase] = (lang, round(hits / total, 4))
        elif cfg['keep_unseen'] and phrase in name2wess and len(phrase) >= cfg['min_len_unseen']:
            targets = name2wess[phrase]
            lang = max(targets, key=lambda w: wess_freq.get(w, 0))
            rules[phrase] = (lang, 0.9)
    return rules


def apply_title_rules(title, rules, max_words=MAX_TITLE_WORDS):
    """When a title names several languages, the most specific one wins.

    Phrases are matched by canon_phrase(), so word order does not matter. A
    longer matching phrase names a narrower language and is preferred:
    "Haitian Creole French" is Haitian, not French. Where two phrases are the
    same length the last-named wins, because uploaders put the specific
    language after the family ("Creole French Haitian" is also Haitian).
    """
    tokens = re.findall(r'\w+', str(title).lower())
    best, answers = None, set()
    for n in range(min(max_words, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            phrase = ' '.join(sorted(tokens[i:i + n]))
            if phrase in rules:
                answers.add(rules[phrase][0])
                if best is None or (n, i + n) > best[:2]:
                    best = (n, i + n, phrase)
    if best is None:
        return None
    # third element: the title named several languages, so the key above had to
    # choose - ASR is consulted on exactly these rows (see predict()).
    return (*rules[best[2]], len(answers) > 1)


# ---------------------------------------------------------------------------
# Tier 3: supervised fastText on channel-prior tokens + historical titles
# ---------------------------------------------------------------------------
def channel_counters(df):
    """channel_id -> Counter(language_id) over labeled history."""
    return {channel: Counter(langs)
            for channel, langs in df.groupby('channel_id')['lang']}


def channel_tokens(counter, own_lang=None, k=2):
    """Prior tokens like '__ch_6464' for the channel's top historical languages.

    own_lang subtracts the current row's own label (leave-one-out), so the model
    cannot read its training answer out of the token at fit time.
    """
    if not counter:
        return '__ch_none'
    counts = counter.copy()
    if own_lang:
        counts[own_lang] -= 1
    top = [lang for lang, n in counts.most_common(k) if n > 0]
    return ' '.join(f'__ch_{lang}' for lang in top) if top else '__ch_none'


def ft_input(title, tokens):
    return f'{tokens} {norm_title(title)}'.strip()


def train_fasttext(df, counters, status):
    import fasttext
    rows = df[df['video_title'].str.strip() != '']
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as handle:
        for channel, title, lang in zip(rows['channel_id'], rows['video_title'], rows['lang']):
            tokens = channel_tokens(counters.get(channel), own_lang=lang)
            handle.write(f'__label__{lang} {ft_input(title, tokens)}\n')
        train_file = handle.name
    status(f'Training fastText on {len(rows)} titles '
           f'({rows["lang"].nunique()} languages)')
    model = fasttext.train_supervised(
        train_file, lr=0.5, epoch=20, wordNgrams=2, minn=2, maxn=5,
        dim=64, loss='softmax', bucket=1_000_000,
        seed=FT_SEED, thread=FT_THREADS, verbose=0)
    try:
        model.quantize(input=train_file, cutoff=200_000, retrain=True, qnorm=True,
                       thread=FT_THREADS, verbose=0)
    except Exception as exc:  # quantization is a size optimization only
        status(f'fastText quantization skipped ({exc})')
    os.unlink(train_file)
    return model


def fasttext_predict(model, inputs):
    """[(language_id | None, prob)] per prepared input line; None when empty."""
    texts = [str(t).replace('\n', ' ').strip() for t in inputs]
    keep = [i for i, t in enumerate(texts) if t and t != '__ch_none']
    out = [(None, 0.0)] * len(texts)
    if keep:
        labels, probs = model.predict([texts[i] for i in keep], k=1)
        for i, label, prob in zip(keep, labels, probs):
            if label:
                out[i] = (label[0].replace('__label__', ''), float(prob[0]))
    return out


# ---------------------------------------------------------------------------
# Tier 4: pretrained lid.176 -> ISO -> WESS
# ---------------------------------------------------------------------------
def get_lid_model():
    import fasttext
    if not LID_MODEL_PATH.exists():
        import urllib.request
        urllib.request.urlretrieve(LID_URL, LID_MODEL_PATH)
    return fasttext.load_model(str(LID_MODEL_PATH))


def build_lid_label_map(lid_model, iso2wess, wess_freq):
    """lid.176 label ('en', 'ceb', ...) -> WESS id, or None when unmappable.

    Ambiguity (an ISO code with several WESS rows) is resolved by historical
    label frequency.
    """
    out = {}
    for raw in lid_model.get_labels():
        code = raw.replace('__label__', '')
        for iso in ISO1_TO_3.get(code, [code] if len(code) == 3 else []):
            if iso in iso2wess:
                out[code] = max(iso2wess[iso], key=lambda w: wess_freq.get(w, 0))
                break
    return out


def lid_predict(model, label_map, titles):
    """[(language_id | None, prob)] per title via pretrained language ID."""
    texts = [norm_title(t) for t in titles]
    keep = [i for i, t in enumerate(texts) if t]
    out = [(None, 0.0)] * len(texts)
    if keep:
        labels, probs = model.predict([texts[i] for i in keep], k=1)
        for i, label, prob in zip(keep, labels, probs):
            if label:
                code = label[0].replace('__label__', '')
                out[i] = (label_map.get(code), float(prob[0]))
    return out


# ---------------------------------------------------------------------------
# Cutoff tuning (most permissive cutoff whose bucket stays >= PRECISION_TARGET)
# ---------------------------------------------------------------------------
# captions.list failures, classified by the API's reason code. The HTTP status
# alone cannot separate them: 'forbidden' (captions not viewable) and
# 'quotaExceeded' are both 403.
ASR_NO_TRACK_REASONS = {'forbidden', 'videoNotFound', 'captionNotFound'}
ASR_RETRY_REASONS = {'rateLimitExceeded', 'userRateLimitExceeded', 'servingLimitExceeded',
                     'internalError', 'backendError', 'backendNotConnected', 'notReady'}
ASR_STOP_REASONS = {'quotaExceeded', 'quotaExceeded402', 'dailyLimitExceeded',
                    'dailyLimitExceeded402', 'dailyLimitExceededUnreg',
                    'variableTermLimitExceeded', 'variableTermExpiredDailyExceeded',
                    'concurrentLimitExceeded', 'keyInvalid', 'keyExpired', 'accessNotConfigured'}
ASR_RETRIES = 2                   # extra attempts for a transient failure
ASR_MAX_CONSECUTIVE_FAILURES = 5  # unrecorded failures in a row that mean an outage


def http_error_reason(exc):
    """The API's reason code ('quotaExceeded', 'forbidden', ...) from an HttpError.

    HttpError.reason is the human-readable message, not the code, so the code is
    read from the response body.
    """
    try:
        error = json.loads(exc.content.decode('utf-8'))['error']
    except (ValueError, KeyError, TypeError, AttributeError):
        return ''
    for item in list(error.get('errors') or []) + list(error.get('details') or []):
        if isinstance(item, dict) and item.get('reason'):
            return item['reason']
    return ''


def asr_languages(video_ids, api_key, status=None, report=None):
    """video_id -> ISO code of the video's automatic captions ('' when it has none).

    YouTube runs speech recognition on most uploads, and the resulting track is
    labelled with the language it heard. That is evidence about the audio, which
    is what the reviewer actually judges - unlike the title, which is often in a
    different language from the film.

    Only definite answers are returned. '' means YouTube answered and there is no
    usable track: none was generated, the captions are not viewable, or the video
    is gone. A video whose lookup failed is left out entirely, because a failure
    is not an answer - recording it as '' would make an exhausted quota look like
    a video without captions, and a cache would never ask again. The run stops at
    the first quota or key error, since every later call would fail the same way.

    Costs 50 quota units per video, so callers should pass only the rows no
    cheaper tier could answer. Pass a dict as `report` to learn how the run ended:
    looked_up, answered, failed and stopped (None, 'quota' or 'outage').
    """
    import time
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    youtube = build('youtube', 'v3', developerKey=api_key)
    out, failed, consecutive, attempted = {}, 0, 0, 0
    tally = {} if report is None else report
    tally.update({'looked_up': 0, 'answered': 0, 'failed': 0, 'stopped': None, 'reason': ''})

    def finish(stopped=None, reason=''):
        tally.update({'looked_up': attempted, 'answered': len(out), 'failed': failed,
                      'stopped': stopped, 'reason': reason})
        return out

    for k, video_id in enumerate(video_ids, 1):
        attempted = k
        error, retryable = '', False
        for attempt in range(ASR_RETRIES + 1):
            try:
                response = youtube.captions().list(part='snippet',
                                                   videoId=video_id).execute()
                tracks = [item['snippet'] for item in response.get('items', [])]
                asr = [t for t in tracks if t.get('trackKind') == 'asr']
                out[video_id] = (asr[0].get('language') or '') if asr else ''
                break
            except HttpError as exc:
                reason = http_error_reason(exc)
                if reason in ASR_NO_TRACK_REASONS:
                    out[video_id] = ''
                    break
                if reason in ASR_STOP_REASONS:
                    if status:
                        status(f'  ASR: stopped at lookup {k} of {len(video_ids)} ({reason}); '
                               f'{len(video_ids) - k + 1} left unrecorded')
                    return finish('quota', reason)
                error = reason or f'HTTP {exc.status_code}'
                retryable = reason in ASR_RETRY_REASONS or exc.status_code >= 500
            except Exception as exc:  # network trouble: timeouts, dropped connections
                error, retryable = type(exc).__name__, True
            if not retryable or attempt == ASR_RETRIES:
                break
            time.sleep(2 ** attempt)
        if video_id in out:
            consecutive = 0
            continue
        failed += 1
        consecutive += 1
        if status:
            status(f'  ASR: {video_id} not looked up ({error}); left unrecorded')
        if consecutive >= ASR_MAX_CONSECUTIVE_FAILURES:
            if status:
                status(f'  ASR: stopped after {consecutive} failures in a row; '
                       f'{len(video_ids) - k} left unrecorded')
            return finish('outage', error)
    if status and failed:
        status(f'  ASR: {failed} of {len(video_ids)} lookups failed and were left unrecorded')
    return finish()

def asr_language_report(asr_iso, iso2wess, wess_freq, truth):
    """Per ASR language: rows, correct, precision, base rate and lift.

    Base rate is how often that language is the truth among all the rows being
    judged, whatever ASR said. Precision alone flatters a language that simply
    dominates the batch - always answering Spanish on a batch that is a quarter
    Spanish scores 25% for free - so lift (precision / base rate) shows how much
    of the precision is the signal rather than the prior.
    """
    judged = len(truth)
    truth_counts = Counter(truth)
    per_lang = defaultdict(lambda: [0, 0, None])
    for iso, true_lang in zip(asr_iso, truth):
        wess = asr_to_wess(iso, iso2wess, wess_freq)
        if wess is None:
            continue
        entry = per_lang[asr_base(iso)]
        entry[0] += int(wess == true_lang)
        entry[1] += 1
        entry[2] = wess
    report = {}
    for code, (hits, n, wess) in per_lang.items():
        precision = hits / n
        base = truth_counts.get(wess, 0) / judged if judged else 0.0
        report[code] = {'n': n, 'correct': hits, 'precision': round(precision, 4),
                        'base_rate': round(base, 4),
                        'lift': round(precision / base, 1) if base else None}
    return report


def tune_asr_languages(asr_iso, iso2wess, wess_freq, truth):
    """Keep only the ASR languages whose mapping to WESS is reliable.

    One ASR label often spans many WESS ids - 'id' covers Djambi, Malaysian,
    North Moluccan Malay and more - and resolving that by history frequency
    guesses wrong far more often than it guesses right. Measuring per language
    keeps the ones where the mapping is safe and drops the ones where it is not,
    instead of judging the signal as a whole.
    """
    report = asr_language_report(asr_iso, iso2wess, wess_freq, truth)
    return {code: r['precision'] for code, r in report.items()
            if r['n'] >= ASR_MIN_PER_LANG and r['precision'] >= ASR_PRECISION}


def load_asr_cache(path=None):
    """video_id -> caption code for every definite answer fetched so far.

    The cache is an append-only JSON-lines log. It holds what YouTube said, never
    a judgement on it: whether a code is trusted is decided at certification, so
    a language certified later can use answers fetched months earlier.
    """
    path = Path(path or ASR_CACHE_PATH)
    cache = {}
    if path.exists():
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:   # a line cut short by a crash mid-write
                    continue
                cache[record['video_id']] = record['code']
    return cache


def append_asr_cache(answers, path=None):
    """Append fetched answers to the cache log."""
    if not answers:
        return
    path = Path(path or ASR_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    # A crash mid-write can leave the last line without its newline. Start a new
    # line, so the torn record is the only one lost.
    torn = False
    if path.exists() and path.stat().st_size:
        with open(path, 'rb') as handle:
            handle.seek(-1, os.SEEK_END)
            torn = handle.read(1) != b'\n'
    with open(path, 'a', encoding='utf-8') as handle:
        if torn:
            handle.write('\n')
        for video_id, code in answers.items():
            handle.write(json.dumps({'video_id': video_id, 'code': code,
                                     'fetched_at': stamp}) + '\n')


def fill_asr_cache(video_ids, cache, api_key=None, limit=0, status=None, path=None, report=None):
    """Look up at most `limit` of these videos that the cache has not seen yet.

    Needs an API key; without one the cache is returned as it is. The budget is
    spent from the front, so callers pass video_ids most important first. Every
    new answer is written to the cache, so a video is only ever paid for once.
    """
    missing = [v for v in dict.fromkeys(video_ids) if v and v not in cache]
    if api_key and missing and limit > 0:
        fetched = asr_languages(missing[:limit], api_key, status, report)
        append_asr_cache(fetched, path)
        cache.update(fetched)
    return cache


def collection_rank(triage, licensed):
    """Lower sorts first: approved, then to review, then no triage, then near-certain N.

    A licensed asset is force-rejected by the verdict model, and a rejected claim
    never needs a language - but these go last, not never: reviewers approved 3.1%
    of the licensed bucket in July, and an approved claim does need one. Rows with
    no triage at all (the daily ingest export carries none) sort ahead of the
    auto-rejected, because an unknown row is likelier to need an answer.
    """
    if licensed or str(triage or '').startswith('AUTO_N'):
        return 3
    return TRIAGE_PRIORITY.get(triage, 2)


def asr_collection_order(df, artifact=None):
    """Video ids of a scored queue, in the order the collector should look them up.

    Rows CHANNEL or TITLE already answer are skipped - except contested titles,
    which ASR may overrule. The rest go claims likely to be approved first
    (AUTO_Y, then REVIEW, then the auto-rejected), most-viewed first within each:
    a language is only used on an approved claim, and approved claims are also
    the rows that certify languages. The auto-rejected and licensed rows still
    come last rather than never, because reviewers overturn a few percent of them.
    Both `triage` and `licensed` are optional: the daily ingest export carries
    neither, and then everything sorts on views alone.
    """
    artifact = artifact or {}
    tiers = artifact.get('tiers', {})
    cmap = artifact.get('channel_map', {}) if 'CHANNEL' in tiers else {}
    rules = ({k: tuple(v) for k, v in artifact.get('title_rules', {}).items()}
             if 'TITLE' in tiers else {})
    ranked = []
    for position, row in enumerate(df.to_dict('records')):
        channel = row.get('channel_id')
        if isinstance(channel, str) and channel in cmap:
            continue
        title = row.get('video_title')
        if rules and isinstance(title, str):
            match = apply_title_rules(title, rules)
            if match and not match[2]:
                continue
        licensed = str(row.get('licensed', '')).strip().lower() in TRUTHY
        views = pandas.to_numeric(row.get('views'), errors='coerce')
        ranked.append((collection_rank(row.get('triage'), licensed),
                       -(0.0 if pandas.isna(views) else float(views)),
                       position, normalize_id(row.get('video_id'))))
    return list(dict.fromkeys(video_id for *_, video_id in sorted(ranked) if video_id))


def write_collector_status(payload, path=None):
    """Write the collector's last-run summary, atomically so a poller never reads half."""
    path = Path(path or ASR_STATUS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def collect_asr(df, api_key, limit, status, artifact=None, path=None, status_path=None):
    """One day of collection: look up the next `limit` uncached videos of a queue."""
    order = asr_collection_order(df, artifact)
    cache = load_asr_cache(path)
    before = sum(1 for v in order if v in cache)
    report = {}
    fill_asr_cache(order, cache, api_key, limit, status, path, report)
    after = sum(1 for v in order if v in cache)
    left = len(order) - after
    write_collector_status({
        'last_run': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'looked_up': report.get('looked_up', 0),
        'added': after - before,
        'failed': report.get('failed', 0),
        'remaining': left,
        'queue_videos_needing_asr': len(order),
        'budget': limit,
        'stopped_reason': report.get('stopped'),
        'stopped_detail': report.get('reason', ''),
    }, status_path)
    status(f'ASR collector: {after - before} answers added; {after} of {len(order)} queue '
           f'videos needing ASR are cached, {left} to go'
           + (f' (~{-(-left // limit)} more day(s) at {limit}/day)' if left and limit else ''))
    return after - before

def asr_base(iso_code):
    """'es-419' -> 'es'. Tuning, the fire count and prediction must key a
    language identically, or a region variant splits its tuning rows and a
    language certified as 'es-419' never matches at prediction time."""
    return str(iso_code).split('-')[0].lower()


def asr_to_wess(iso_code, iso2wess, wess_freq):
    """ASR's BCP-47 label -> WESS id, ambiguity resolved by history frequency."""
    if not iso_code:
        return None
    base = asr_base(iso_code)
    for iso in ISO1_TO_3.get(base, [base] if len(base) == 3 else []):
        if iso in iso2wess:
            return max(iso2wess[iso], key=lambda w: wess_freq.get(w, 0))
    return None


def tune_prob_cutoff(pred_lang, pred_prob, truth):
    grid = np.concatenate([np.arange(0.30, 0.99, 0.01),        # coarse
                           np.arange(0.99, 0.99991, 0.0005)])  # softmax mass sits near 1
    for cutoff in np.round(grid, 4):
        fired = [(p, t) for p, prob, t in zip(pred_lang, pred_prob, truth)
                 if p is not None and prob >= cutoff]
        if len(fired) < MIN_BUCKET_N:
            break
        precision = sum(p == t for p, t in fired) / len(fired)
        if precision >= PRECISION_TARGET:
            return float(cutoff), precision, len(fired)
    return None


def measure(pairs):
    """[(pred, truth)] -> (precision, n) over fired rows."""
    fired = [(p, t) for p, t in pairs if p is not None]
    if not fired:
        return 0.0, 0
    return sum(p == t for p, t in fired) / len(fired), len(fired)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def in_calibration_half(video_id):
    """Stable 50/50 split of a reviewed batch: calibration vs held-out half."""
    return int(hashlib.md5(str(video_id).encode()).hexdigest(), 16) % 2 == 0


def train(history_path, status, exclude_video_ids=(), calib_df=None, asr_key=None,
          asr_limit=ASR_DEFAULT_LIMIT):
    """Fit signals on labeled history and tune per-tier cutoffs.

    Cutoffs are tuned on the calibration half of a reviewed batch when one is
    supplied (--eval-labels; the other half stays held out for scoring). This
    matches how the verdict model's triage cutoffs are recalibrated: a temporal
    holdout of history is easier than real monthly leftovers. Without a
    reviewed batch it falls back to the most recent VAL_DAYS of history.
    """
    wess2name, name2wess, iso2wess = load_mapping()

    status(f'Loading labeled history from {history_path}')
    df = load_history(history_path, exclude_video_ids)
    status(f'{len(df)} labeled claims, {df["lang"].nunique()} languages')

    if calib_df is not None and len(calib_df) >= 4 * MIN_BUCKET_N:
        fit_df = df  # signals use all history; tuning data is external
        tune_df = calib_df[calib_df['video_id'].map(in_calibration_half)]
        status(f'Tuning cutoffs on calibration half of the reviewed batch '
               f'({len(tune_df)} of {len(calib_df)} labeled claims)')
    else:
        holdout_start = df['created'].max() - pandas.Timedelta(days=VAL_DAYS)
        val_mask = df['created'] >= holdout_start
        if val_mask.sum() < 500:  # tiny export; fall back to a 20% temporal split
            val_mask = df['created'] >= df['created'].quantile(0.8)
        fit_df, tune_df = df[~val_mask], df[val_mask]
        status(f'Tuning cutoffs on temporal holdout: fit {len(fit_df)}, '
               f'holdout {len(tune_df)} (last {VAL_DAYS} days)')

    wess_freq = Counter(fit_df['lang'])
    counters = channel_counters(fit_df)
    truth = tune_df['lang'].tolist()
    tiers = {}
    # Why each tier that failed its gate was dropped. A disabled tier used to
    # vanish from the artifact with no trace, which makes a tier that stopped
    # firing indistinguishable from one that was never built.
    disabled = {}

    # ---- CHANNEL: smallest min_count whose tuning precision meets target
    for min_count in (1, 2, 3, 5, 10):
        cmap = build_channel_map(fit_df, min_count)
        prec, n = measure([(cmap.get(c), t) for c, t in
                           zip(tune_df['channel_id'], truth)])
        status(f'  CHANNEL min_count={min_count}: precision {prec:.3f} on {n}')
        if n < MIN_BUCKET_N:
            disabled['CHANNEL'] = {'reason': f'fired {n} < {MIN_BUCKET_N} at min_count={min_count}',
                                   'val_precision': round(prec, 4), 'val_n': n}
            break
        if prec >= PRECISION_TARGET:
            tiers['CHANNEL'] = {'min_count': min_count, 'val_precision': round(prec, 4), 'val_n': n}
            disabled.pop('CHANNEL', None)   # a looser min_count failing is not a failed tier
            break
        disabled['CHANNEL'] = {'reason': f'best precision {prec:.3f} < {PRECISION_TARGET} '
                                         f'(min_count={min_count}, fired {n})',
                               'val_precision': round(prec, 4), 'val_n': n}

    # ---- TITLE: among configs meeting the target, widest coverage wins
    title_cfgs = [
        {'min_n': 5, 'min_prec': 0.95, 'keep_unseen': False, 'min_len_unseen': 5},
        {'min_n': 3, 'min_prec': 0.9, 'keep_unseen': False, 'min_len_unseen': 5},
        {'min_n': 3, 'min_prec': 0.9, 'keep_unseen': True, 'min_len_unseen': 5},
    ]
    best_title = None
    last_title_prec, last_title_n = 0.0, 0
    for cfg in title_cfgs:
        rules = build_title_rules(name2wess, wess_freq, fit_df, cfg)
        prec, n = measure([(match[0] if (match := apply_title_rules(t, rules)) else None, t_lang)
                           for t, t_lang in zip(tune_df['video_title'], truth)])
        status(f'  TITLE {cfg}: {len(rules)} rules, precision {prec:.3f} on {n}')
        if n > last_title_n:
            last_title_prec, last_title_n = prec, n
        if n >= MIN_BUCKET_N and prec >= PRECISION_TARGET:
            if best_title is None or n > best_title['val_n']:
                best_title = {'cfg': cfg, 'val_precision': round(prec, 4), 'val_n': n}
    if best_title:
        tiers['TITLE'] = best_title
        disabled.pop('TITLE', None)
    else:
        disabled['TITLE'] = {'reason': f'no config reached {PRECISION_TARGET} on >= {MIN_BUCKET_N} '
                                       f'rows (best: {last_title_prec:.3f} on {last_title_n})',
                             'val_precision': round(last_title_prec, 4), 'val_n': last_title_n}

    # ---- ASR: trust the audio, but only for languages whose mapping is safe.
    # Certification reads the caption cache the collector fills during the month,
    # so it spends no quota. With --asr it may also look up uncached tuning rows,
    # within --asr-limit.
    asr_cache = load_asr_cache()
    ids = [normalize_id(v) for v in tune_df['video_id']]
    if asr_key or any(v in asr_cache for v in ids):
        fill_asr_cache(ids, asr_cache, asr_key, asr_limit, status)
        cached = sum(1 for v in ids if v in asr_cache)
        status(f'  ASR: {cached} of {len(ids)} tuning rows have a cached caption lookup')
        asr_iso = [asr_cache.get(v, '') for v in ids]
        report = asr_language_report(asr_iso, iso2wess, wess_freq, truth)
        for code, r in sorted(report.items(), key=lambda kv: -kv[1]['n'])[:8]:
            status(f'    {code}: {r["correct"]}/{r["n"]} = {r["precision"]:.3f}, '
                   f'base rate {r["base_rate"]:.3f}, lift {r["lift"]}')
        keep = tune_asr_languages(asr_iso, iso2wess, wess_freq, truth)
        fired = sum(1 for iso in asr_iso if asr_base(iso) in keep)
        if keep and fired >= MIN_BUCKET_N:
            tiers['ASR'] = {'languages': keep, 'report': report}
            status(f'  ASR: trusting {len(keep)} language(s) {sorted(keep)}, '
                   f'fires on {fired} of {len(tune_df)}')
        else:
            disabled['ASR'] = {'reason': f'no ISO code reached {ASR_MIN_PER_LANG} rows at '
                                         f'{ASR_PRECISION} precision ({cached} of {len(ids)} '
                                         f'tuning rows cached)',
                               'val_n': fired, 'report': report}
            status(f'  ASR: no language met {ASR_PRECISION} on >= {ASR_MIN_PER_LANG} '
                   f'rows (or too few fires); tier disabled')

    # ---- FASTTEXT supervised (channel-prior tokens + title): tune cutoff
    ft_model = train_fasttext(fit_df, counters, status)
    inputs = [ft_input(t, channel_tokens(counters.get(c)))
              for t, c in zip(tune_df['video_title'], tune_df['channel_id'])]
    preds = fasttext_predict(ft_model, inputs)
    tuned = tune_prob_cutoff([p for p, _ in preds], [pr for _, pr in preds], truth)
    if tuned:
        cutoff, prec, n = tuned
        tiers['FASTTEXT'] = {'cutoff': cutoff, 'val_precision': round(prec, 4), 'val_n': n}
        status(f'  FASTTEXT: cutoff {cutoff:.2f}, precision {prec:.3f} on {n}')
    else:
        disabled['FASTTEXT'] = {'reason': f'no cutoff on the grid reached {PRECISION_TARGET} '
                                          f'on >= {MIN_BUCKET_N} rows'}
        status('  FASTTEXT: no cutoff met the precision target; tier disabled')

    # ---- LID pretrained: tune cutoff
    try:
        lid_model = get_lid_model()
        lid_map = build_lid_label_map(lid_model, iso2wess, wess_freq)
        preds = lid_predict(lid_model, lid_map, tune_df['video_title'])
        tuned = tune_prob_cutoff([p for p, _ in preds], [pr for _, pr in preds], truth)
        if tuned:
            cutoff, prec, n = tuned
            tiers['LID'] = {'cutoff': cutoff, 'val_precision': round(prec, 4), 'val_n': n}
            status(f'  LID: cutoff {cutoff:.2f}, precision {prec:.3f} on {n}')
        else:
            disabled['LID'] = {'reason': f'no cutoff on the grid reached {PRECISION_TARGET} '
                                         f'on >= {MIN_BUCKET_N} rows'}
            status('  LID: no cutoff met the precision target; tier disabled')
    except Exception as exc:
        disabled['LID'] = {'reason': f'model unavailable: {exc}'}
        status(f'  LID tier disabled ({exc})')

    if fit_df is not df:
        status('Refitting signals on all labeled history')
    wess_freq_all = Counter(df['lang'])
    counters_all = channel_counters(df)
    artifact = {
        'tiers': tiers,
        'disabled_tiers': disabled,
        'wess2name': wess2name,
        'channel_tokens': {ch: channel_tokens(ctr) for ch, ctr in counters_all.items()},
        'metadata': {
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'history': str(history_path),
            'labeled_rows': int(len(df)),
            'languages': int(df['lang'].nunique()),
            'excluded_video_ids': len(exclude_video_ids),
            'calibrated_on_reviewed_batch': calib_df is not None,
        },
    }
    if 'CHANNEL' in tiers:
        artifact['channel_map'] = build_channel_map(df, tiers['CHANNEL']['min_count'])
    if 'TITLE' in tiers:
        rules = build_title_rules(name2wess, wess_freq_all, df, tiers['TITLE']['cfg'])
        artifact['title_rules'] = {k: list(v) for k, v in rules.items()}
    if 'LID' in tiers:
        artifact['lid_label_map'] = build_lid_label_map(get_lid_model(), iso2wess, wess_freq_all)
    if 'ASR' in tiers:
        # the tier resolves ISO -> WESS at predict time, so it needs both tables
        artifact['iso2wess'] = {k: sorted(v) for k, v in iso2wess.items()}
        artifact['wess_freq'] = {k: int(v) for k, v in wess_freq_all.items()}

    final_ft = None
    if 'FASTTEXT' in tiers:
        final_ft = ft_model if fit_df is df else train_fasttext(df, counters_all, status)
        final_ft.save_model(str(FT_MODEL_PATH))
    ARTIFACT_PATH.write_text(json.dumps(artifact))
    status(f'Saved {ARTIFACT_PATH.name}'
           + (f' + {FT_MODEL_PATH.name}' if final_ft else ''))
    status(f'Tiers live: {" -> ".join(t for t in CASCADE if t in tiers) or "none"}')
    for tier in CASCADE:
        if tier in disabled:
            status(f'  {tier} disabled: {disabled[tier]["reason"]}')
    return artifact


def load_artifact():
    return json.loads(ARTIFACT_PATH.read_text())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(df, artifact, status, asr_key=None, asr_limit=ASR_DEFAULT_LIMIT):
    tiers = artifact['tiers']
    wess2name = artifact['wess2name']
    n = len(df)
    lang = [None] * n
    source = ['REVIEW'] * n
    conf = [np.nan] * n
    contested = []   # TITLE rows whose title named more than one language
    off = artifact.get('disabled_tiers', {})
    if off:
        status('Tiers not in this artifact: '
               + '; '.join(f'{t} ({off[t]["reason"]})' for t in CASCADE if t in off))

    def assign(i, language, tier, confidence):
        lang[i], source[i], conf[i] = language, tier, round(float(confidence), 4)

    if 'CHANNEL' in tiers:
        cmap = artifact['channel_map']
        for i, channel in enumerate(df['channel_id'].fillna('')):
            if channel in cmap:
                assign(i, cmap[channel], 'CHANNEL', 1.0)

    if 'TITLE' in tiers:
        rules = {k: tuple(v) for k, v in artifact['title_rules'].items()}
        for i, title in enumerate(df['video_title'].fillna('')):
            if source[i] == 'REVIEW':
                match = apply_title_rules(title, rules)
                if match:
                    assign(i, match[0], 'TITLE', match[1])
                    if match[2]:
                        contested.append(i)

    pending = [i for i in range(n) if source[i] == 'REVIEW']
    # ASR arbitrates the handful of titles that named several languages, and
    # otherwise answers rows no cheaper tier could. Answers come from the caption
    # cache the collector fills; --asr only adds live lookups for rows it has not
    # reached, within --asr-limit, contested titles first.
    targets = contested + pending
    if 'ASR' in tiers and targets:
        keep = tiers['ASR']['languages']
        iso2wess = artifact['iso2wess']
        wess_freq = artifact['wess_freq']
        ids = [normalize_id(df['video_id'].iloc[i]) for i in targets]
        asr_cache = load_asr_cache()
        cached_before = sum(1 for v in ids if v in asr_cache)
        fill_asr_cache(ids, asr_cache, asr_key, asr_limit, status)
        cached = sum(1 for v in ids if v in asr_cache)
        status(f'  ASR: {cached} of {len(ids)} rows have a caption lookup '
               f'({cached_before} cached, {cached - cached_before} looked up now)')
        for i, video_id in zip(targets, ids):
            iso = asr_cache.get(video_id, '')
            base = asr_base(iso)
            if base in keep:
                wess = asr_to_wess(iso, iso2wess, wess_freq)
                if wess is not None:
                    assign(i, wess, 'ASR', keep[base])

    pending = [i for i in range(n) if source[i] == 'REVIEW']
    if 'FASTTEXT' in tiers and pending:
        import fasttext
        model = fasttext.load_model(str(FT_MODEL_PATH))
        tokens_map = artifact.get('channel_tokens', {})
        titles = df['video_title'].fillna('').tolist()
        channels = df['channel_id'].fillna('').tolist()
        cutoff = tiers['FASTTEXT']['cutoff']
        inputs = [ft_input(titles[i], tokens_map.get(channels[i], '__ch_none'))
                  for i in pending]
        for i, (language, prob) in zip(pending, fasttext_predict(model, inputs)):
            if language is not None and prob >= cutoff:
                assign(i, language, 'FASTTEXT', prob)

    pending = [i for i in range(n) if source[i] == 'REVIEW']
    if 'LID' in tiers and pending:
        lid_model = get_lid_model()
        lid_map = artifact['lid_label_map']
        titles = df['video_title'].fillna('').tolist()
        cutoff = tiers['LID']['cutoff']
        for i, (language, prob) in zip(pending, lid_predict(lid_model, lid_map, [titles[i] for i in pending])):
            if language is not None and prob >= cutoff:
                assign(i, language, 'LID', prob)

    out = df.copy()
    out['predicted_language_id'] = lang
    out['predicted_language_name'] = [wess2name.get(l, '') if l else '' for l in lang]
    out['language_source'] = source
    out['language_confidence'] = conf
    fired = sum(1 for s in source if s != 'REVIEW')
    status(f'Predicted language for {fired}/{n} claims '
           f'({Counter(source).most_common()})')
    return out


# ---------------------------------------------------------------------------
# Evaluation against a completed monthly sheet
# ---------------------------------------------------------------------------
def label_paths(value):
    """One or more reviewed verdict sheets -> list of paths.

    Accepts a path, a comma-separated string of paths, or a list of either: the
    monthly review arrives as two sheets (MCN and JFM).
    """
    if not value:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    return [part.strip() for item in items for part in str(item).split(',') if part.strip()]


def read_verdict_sheet(path):
    """One raw reviewed verdict sheet: video_id and language_id only, headers stripped.

    Built for the files the claims pipeline uploads as-is: columns are found by
    name in any order, header names are stripped (a July sheet has ' wave'),
    trailing empty columns and malformed rows are ignored, and the path needn't
    end in .csv.
    """
    sheet = pandas.read_csv(path, dtype=str, encoding='utf-8-sig', on_bad_lines='skip',
                            usecols=lambda column: str(column).strip() in ('video_id', 'language_id'))
    sheet.columns = [str(column).strip() for column in sheet.columns]
    return sheet


def read_labels(labels_path, status=None):
    """video_id -> WESS language from one or more reviewed verdict sheets.

    A sheet without video_id or language_id contributes no labels rather than an
    error. Rows without a language (no verdict, or not approved) are dropped.
    """
    frames = []
    for path in label_paths(labels_path):
        sheet = read_verdict_sheet(path)
        if not {'video_id', 'language_id'} <= set(sheet.columns):
            if status:
                status(f'  labels: {Path(path).name} has no video_id/language_id columns - skipped')
            continue
        frames.append(sheet[['video_id', 'language_id']])
    if not frames:
        return pandas.DataFrame(columns=['video_id', 'true_lang'])
    labels = pandas.concat(frames, ignore_index=True)
    labels['true_lang'] = labels['language_id'].map(norm_lang)
    labels = labels[labels['true_lang'].notna() & labels['video_id'].notna()]
    return labels.drop_duplicates('video_id', keep='last')[['video_id', 'true_lang']]

def train_from_exports(history_path, eval_labels, status):
    """Monthly retrain from the claims pipeline's own exports.

    history_path is the run's all_claims.csv and eval_labels the reviewed verdict
    sheet(s). Tuning rows take their titles and channels from all_claims, not from
    the queue about to be predicted: the reviewed batch and the new queue barely
    overlap (~2% of videos), so joining the labels onto the queue would leave
    almost nothing to tune on. ASR is certified from the caption cache, with no
    API calls.
    """
    import csv
    import sys
    labels = read_labels(eval_labels, status)
    if labels.empty:
        raise ValueError('no reviewed claims with a language in the verdict sheets')
    wanted = {normalize_id(v): lang for v, lang in zip(labels['video_id'], labels['true_lang'])}
    csv.field_size_limit(sys.maxsize)
    rows, seen = [], set()
    with open(history_path, newline='', encoding='utf-8', errors='replace') as handle:
        for row in csv.DictReader(handle):
            video_id = normalize_id(row.get('video_id') or '')
            if video_id in wanted and video_id not in seen:
                seen.add(video_id)
                rows.append({'video_id': video_id, 'video_title': row.get('video_title') or '',
                             'channel_id': row.get('channel_id') or '', 'lang': wanted[video_id]})
    calib_df = pandas.DataFrame(rows, columns=['video_id', 'video_title', 'channel_id', 'lang'])
    status(f'Retraining languages: {len(calib_df):,} of {len(wanted):,} reviewed claims found in '
           f'{Path(history_path).name} ({100 * len(calib_df) / max(len(wanted), 1):.1f}%)')
    exclude = tuple(set(labels['video_id']) | set(wanted))
    return train(history_path, status, exclude_video_ids=exclude, calib_df=calib_df)


def report(merged, status):
    rows = []
    for tier in CASCADE + ['REVIEW']:
        subset = merged[merged['language_source'] == tier]
        if not len(subset):
            continue
        correct = int((subset['predicted_language_id'] == subset['true_lang']).sum())
        rows.append((tier, len(subset), correct))
    fired = merged[merged['language_source'] != 'REVIEW']
    total_correct = int((fired['predicted_language_id'] == fired['true_lang']).sum())

    width = max((len(t) for t, _, _ in rows), default=8)
    status(f'  {"tier".ljust(width)}  coverage           accuracy')
    for tier, n, correct in rows:
        cov = f'{n}/{len(merged)} ({n / len(merged):.1%})'
        acc = '-' if tier == 'REVIEW' else f'{correct}/{n} = {correct / n:.1%}'
        status(f'  {tier.ljust(width)}  {cov.ljust(17)}  {acc}')
    if len(fired):
        status(f'  {"TOTAL".ljust(width)}  {len(fired)}/{len(merged)} '
               f'({len(fired) / len(merged):.1%})  '
               f'{total_correct}/{len(fired)} = {total_correct / len(fired):.1%}')


def evaluate(out_df, labels_path, status, calibrated=False):
    merged = out_df.merge(read_labels(labels_path), on='video_id', how='inner')
    status(f'\nEvaluation against {labels_path} '
           f'({len(merged)} claims with a human language label):')
    if calibrated:
        holdout = merged[~merged['video_id'].map(in_calibration_half)]
        status(f'\nHeld-out half (never used for tuning; {len(holdout)} claims):')
        report(holdout, status)
        status(f'\nFull reviewed batch (includes the calibration half):')
    report(merged, status)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(args, status_callback=print):
    status = status_callback

    asr_limit = getattr(args, 'asr_limit', None)
    if getattr(args, 'collect_asr', False):
        # Collector mode: one day of caption lookups for a scored queue, then exit.
        from helpers import load_env
        load_env(['YT_API_KEY'])
        queue = pandas.read_csv(args.prediction_input, low_memory=False)
        artifact = load_artifact() if ARTIFACT_PATH.exists() else None
        collect_asr(queue, os.environ['YT_API_KEY'],
                    ASR_DAILY_LIMIT if asr_limit is None else asr_limit, status, artifact)
        return
    if asr_limit is None:
        asr_limit = ASR_DEFAULT_LIMIT

    # Live caption lookups cost 50 quota units per video, so they are opt-in.
    # Without --asr the ASR tier still answers from the collector's cache.
    asr_key = None
    if getattr(args, 'asr', False):
        from helpers import load_env
        load_env(['YT_API_KEY'])
        asr_key = os.environ['YT_API_KEY']

    df = pandas.read_csv(args.prediction_input, low_memory=False)

    exclude = ()
    if args.eval_labels:  # keep evaluation honest if history already has these
        exclude = tuple(pandas.concat(
            [read_verdict_sheet(path).get('video_id', pandas.Series(dtype=str))
             for path in label_paths(args.eval_labels)]).dropna())

    calibrated = False
    if ARTIFACT_PATH.exists():
        status(f'Loading cached artifact from {ARTIFACT_PATH}')
        artifact = load_artifact()
    else:
        if not args.history:
            raise ValueError(f'--history is required to create {ARTIFACT_PATH}')
        calib_df = None
        if args.eval_labels:
            calib_df = df.merge(read_labels(args.eval_labels), on='video_id', how='inner')
            calib_df = calib_df.rename(columns={'true_lang': 'lang'})
            calib_df['video_title'] = calib_df['video_title'].fillna('')
            calibrated = len(calib_df) >= 4 * MIN_BUCKET_N
        artifact = train(args.history, status, exclude_video_ids=exclude,
                         calib_df=calib_df, asr_key=asr_key,
                         asr_limit=asr_limit)

    out = predict(df, artifact, status, asr_key=asr_key,
                  asr_limit=asr_limit)
    out.to_csv(args.prediction_output, index=False)
    status(f'Saved predictions to {args.prediction_output}')

    if args.eval_labels:
        evaluate(out, args.eval_labels, status, calibrated=calibrated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict WESS language_id for unprocessed claims')
    parser.add_argument('--prediction-input', required=True, help='Unprocessed claims CSV')
    parser.add_argument('--history', default=None,
                        help='all_claims export with language_id (required when no cached artifact)')
    parser.add_argument('--eval-labels', default=None,
                        help='Completed monthly sheet (video_id + language_id) to score against')
    parser.add_argument('--asr-limit', type=int, default=None,
                        help=f'Live captions.list lookups: per phase (tuning, prediction) with --asr, '
                             f'default {ASR_DEFAULT_LIMIT}; per run with --collect-asr, default '
                             f'{ASR_DAILY_LIMIT}. 50 quota units each, 10,000/day shared with production.')
    parser.add_argument('--asr', action='store_true',
                        help='Also make live caption lookups for rows the cache lacks (50 quota '
                             'units each; needs YT_API_KEY). Without it, ASR answers from the cache only.')
    parser.add_argument('--collect-asr', action='store_true',
                        help='Collector mode: look up the next --asr-limit uncached videos of '
                             '--prediction-input (a scored queue, ideally with triage and views) into '
                             'data/asr_cache.jsonl, then exit. Run once a day. Needs YT_API_KEY.')
    parser.add_argument('--prediction-output',
                        default=f'wess_predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv',
                        help='Output CSV')
    main(parser.parse_args())

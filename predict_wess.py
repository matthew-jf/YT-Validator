"""Predict WESS language_id for unprocessed claims.

Precision-first cascade, each tier only fires above a cutoff tuned on a
temporal holdout of labeled history (same triage philosophy as pipeline.py):

  1. CHANNEL  - the channel's historical claims all carry one language_id
  2. TITLE    - the title contains a validated language-name rule
                (Anglicized names from sheets_language_families.csv plus
                native-name aliases mined from history, e.g. "bahasa melayu jambi")
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
from datetime import datetime
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
CASCADE = ['CHANNEL', 'TITLE', 'FASTTEXT', 'LID']

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


def title_phrases(title, max_words=3):
    """All 1..max_words word n-grams of the lowercased title, longest first."""
    tokens = re.findall(r'\w+', str(title).lower())
    for n in range(min(max_words, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            yield ' '.join(tokens[i:i + n])


def load_mapping():
    """sheets_language_families.csv -> (wess2name, name2wess, iso2wess)."""
    sheet = pandas.read_csv(SHEETS_PATH, dtype=str).fillna('')
    wess2name, name2wess, iso2wess = {}, defaultdict(set), defaultdict(set)
    for _, row in sheet.iterrows():
        wess = norm_lang(row['WESS_LAN_num'])
        if not wess:
            continue
        name = row['Anglicized_name'].strip()
        iso = row['ISO_lang'].strip().lower()
        wess2name.setdefault(wess, name)
        if name:  # a handful of rows have a blank name; they'd match everything
            name2wess[name.lower()].add(wess)
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
                phrases.add('bahasa ' + ' '.join(words[:n]))
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


def apply_title_rules(title, rules):
    """Longest matching phrase wins (n-grams are generated longest first)."""
    for phrase in title_phrases(title):
        if phrase in rules:
            return rules[phrase]
    return None


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
        thread=max(1, (os.cpu_count() or 4) - 1), verbose=0)
    try:
        model.quantize(input=train_file, cutoff=200_000, retrain=True, qnorm=True, verbose=0)
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


def train(history_path, status, exclude_video_ids=(), calib_df=None):
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

    # ---- CHANNEL: smallest min_count whose tuning precision meets target
    for min_count in (1, 2, 3, 5, 10):
        cmap = build_channel_map(fit_df, min_count)
        prec, n = measure([(cmap.get(c), t) for c, t in
                           zip(tune_df['channel_id'], truth)])
        status(f'  CHANNEL min_count={min_count}: precision {prec:.3f} on {n}')
        if n < MIN_BUCKET_N:
            break
        if prec >= PRECISION_TARGET:
            tiers['CHANNEL'] = {'min_count': min_count, 'val_precision': round(prec, 4), 'val_n': n}
            break

    # ---- TITLE: among configs meeting the target, widest coverage wins
    title_cfgs = [
        {'min_n': 5, 'min_prec': 0.95, 'keep_unseen': False, 'min_len_unseen': 5},
        {'min_n': 3, 'min_prec': 0.9, 'keep_unseen': False, 'min_len_unseen': 5},
        {'min_n': 3, 'min_prec': 0.9, 'keep_unseen': True, 'min_len_unseen': 5},
    ]
    best_title = None
    for cfg in title_cfgs:
        rules = build_title_rules(name2wess, wess_freq, fit_df, cfg)
        prec, n = measure([(match[0] if (match := apply_title_rules(t, rules)) else None, t_lang)
                           for t, t_lang in zip(tune_df['video_title'], truth)])
        status(f'  TITLE {cfg}: {len(rules)} rules, precision {prec:.3f} on {n}')
        if n >= MIN_BUCKET_N and prec >= PRECISION_TARGET:
            if best_title is None or n > best_title['val_n']:
                best_title = {'cfg': cfg, 'val_precision': round(prec, 4), 'val_n': n}
    if best_title:
        tiers['TITLE'] = best_title

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
            status('  LID: no cutoff met the precision target; tier disabled')
    except Exception as exc:
        status(f'  LID tier disabled ({exc})')

    if fit_df is not df:
        status('Refitting signals on all labeled history')
    wess_freq_all = Counter(df['lang'])
    counters_all = channel_counters(df)
    artifact = {
        'tiers': tiers,
        'wess2name': wess2name,
        'channel_tokens': {ch: channel_tokens(ctr) for ch, ctr in counters_all.items()},
        'metadata': {
            'trained_at': datetime.now().isoformat(timespec='seconds'),
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

    final_ft = None
    if 'FASTTEXT' in tiers:
        final_ft = ft_model if fit_df is df else train_fasttext(df, counters_all, status)
        final_ft.save_model(str(FT_MODEL_PATH))
    ARTIFACT_PATH.write_text(json.dumps(artifact))
    status(f'Saved {ARTIFACT_PATH.name}'
           + (f' + {FT_MODEL_PATH.name}' if final_ft else ''))
    return artifact


def load_artifact():
    return json.loads(ARTIFACT_PATH.read_text())


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(df, artifact, status):
    tiers = artifact['tiers']
    wess2name = artifact['wess2name']
    n = len(df)
    lang = [None] * n
    source = ['REVIEW'] * n
    conf = [np.nan] * n

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
def read_labels(labels_path):
    labels = pandas.read_csv(labels_path, usecols=['video_id', 'language_id'], dtype=str)
    labels['true_lang'] = labels['language_id'].map(norm_lang)
    return labels[labels['true_lang'].notna()][['video_id', 'true_lang']]


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

    df = pandas.read_csv(args.prediction_input, low_memory=False)

    exclude = ()
    if args.eval_labels:  # keep evaluation honest if history already has these
        exclude = tuple(pandas.read_csv(args.eval_labels, usecols=['video_id'],
                                        dtype=str)['video_id'].dropna())

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
                         calib_df=calib_df)

    out = predict(df, artifact, status)
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
    parser.add_argument('--prediction-output',
                        default=f'wess_predictions_{datetime.now().strftime("%Y%m%d%H%M")}.csv',
                        help='Output CSV')
    main(parser.parse_args())

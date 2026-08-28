import argparse
import os
import re
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from helpers import load_env

# Suppress warnings
warnings.filterwarnings('ignore')

# Resolve bundled files relative to this file so the CLI works from any cwd
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model.joblib'
LICENSED_PATH = BASE_DIR / 'Licensed.csv'
ASSET_MEDIA_PATH = BASE_DIR / 'assets_single_media_component.csv'
LANG_SHEET_PATH = BASE_DIR / 'sheets_language_families.csv'
LID_MODEL_PATH = BASE_DIR / 'lid.176.ftz'

RANDOM_STATE = 42
TRAIN_START = pandas.Timestamp('2022-01-01')  # recency window that validated best
HOLDOUT_DAYS = 150          # most recent labeled claims, used to tune thresholds
BUCKET_TARGET = 0.95        # per-bucket accuracy target for triage cutoffs
MIN_BUCKET_N = 50
LANG_MIN_PROB = 0.30        # blank predicted_lang below this lid confidence

# The artifact is deserialised once per process rather than once per request, and
# app.py warms it at startup so a corrupt model fails the service immediately
# instead of surfacing inside a background task.
_ARTIFACT = None
_MODEL_LOAD_SECONDS = None
_WARM_SECONDS = None
_LOAD_LOCK = threading.Lock()

# One row exercising every raw column build_features() reads, so warming runs the
# real encode + predict path rather than just touching the file.
_WARM_PROBE = {
    'views': 1, 'matching_duration': 60.0, 'longest_match': 60.0,
    'video_duration_sec': 60.0, 'video_title': 'warmup',
    'claim_created_date': '2026-01-02', 'video_upload_date': '2026-01-01',
    'claim_type': 'AUDIOVISUAL', 'claim_origin': 'UGC', 'content_type': 'VIDEO_MATCH',
    'claim_report_source': 'warmup', 'asset_labels': 'warmup',
    'asset_id': 'warmup', 'custom_id': 'warmup',
    'reference_id': 'warmup', 'channel_id': 'warmup',
}


def peak_rss_mb():
    """Process high-water memory mark, in MB. ru_maxrss is bytes on macOS, KB on Linux."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1048576 if sys.platform == 'darwin' else 1024), 1)
    except Exception:
        return None


def get_artifact():
    """Return the model artifact, deserialising it once per process."""
    global _ARTIFACT, _MODEL_LOAD_SECONDS
    if _ARTIFACT is None:
        with _LOAD_LOCK:
            if _ARTIFACT is None:  # another thread may have won the race
                started = time.time()
                artifact = joblib.load(MODEL_PATH)
                _MODEL_LOAD_SECONDS = time.time() - started
                _ARTIFACT = artifact
    return _ARTIFACT


def warm_model():
    """Load the artifact and score one row, so a corrupt model fails loudly here.

    Raises FileNotFoundError when no artifact exists yet -- that is a valid
    state, since /predict can still be called with training_data to build one.
    """
    global _WARM_SECONDS
    artifact = get_artifact()
    started = time.time()
    artifact['model'].predict_proba(build_features(pandas.DataFrame([_WARM_PROBE]))[FEATURES])
    _WARM_SECONDS = time.time() - started
    return _WARM_SECONDS


def model_info():
    """Describe the loaded model, for /health and deploy verification."""
    meta = (_ARTIFACT or {}).get('metadata', {})
    return {
        'model_path': str(MODEL_PATH),
        'loaded': _ARTIFACT is not None,
        'warm': _WARM_SECONDS is not None,
        'threshold': (_ARTIFACT or {}).get('threshold'),
        't_low': (_ARTIFACT or {}).get('t_low'),
        't_high': (_ARTIFACT or {}).get('t_high'),
        'trained_at': meta.get('trained_at'),
        'train_rows': meta.get('train_rows'),
        'holdout_auc': meta.get('holdout_auc'),
        'load_seconds': round(_MODEL_LOAD_SECONDS, 2) if _MODEL_LOAD_SECONDS else None,
        'warm_seconds': round(_WARM_SECONDS, 2) if _WARM_SECONDS else None,
        'peak_rss_mb': peak_rss_mb(),
    }


# Claims with these no_codes are excluded from training: their verdicts are
# rule-driven (licensed asset, video unavailable, ...) and are reproduced by
# rules at predict time, so the model should only learn content-based verdicts.
excluded_codes = ['L', 'V', 'N', 'X']

NUMERIC = [
    'views_log', 'matching_duration', 'longest_match', 'video_duration_sec',
    'match_ratio', 'longest_ratio', 'match_per_longest',
    'upload_to_claim_days', 'claim_year', 'claim_month', 'title_len',
]
LOW_CARD = ['claim_type', 'claim_origin', 'content_type', 'claim_report_source', 'asset_labels']
HIGH_CARD = ['asset_id', 'custom_id', 'reference_id', 'channel_id']
FEATURES = NUMERIC + LOW_CARD + HIGH_CARD


# ---------------------------------------------------------------------------
# Features / model
# ---------------------------------------------------------------------------
def build_features(df):
    out = pandas.DataFrame(index=df.index)
    created = pandas.to_datetime(df['claim_created_date'], format='mixed', errors='coerce')
    uploaded = pandas.to_datetime(df['video_upload_date'], format='mixed', errors='coerce')

    dur = df['video_duration_sec'].fillna(0).astype(float)
    match = df['matching_duration'].fillna(0).astype(float)
    longest = df['longest_match'].fillna(0).astype(float)

    out['views_log'] = np.log1p(df['views'].fillna(0).astype(float))
    out['matching_duration'] = match
    out['longest_match'] = longest
    out['video_duration_sec'] = dur
    out['match_ratio'] = match / dur.clip(lower=1)
    out['longest_ratio'] = longest / dur.clip(lower=1)
    out['match_per_longest'] = match / longest.clip(lower=1)
    out['upload_to_claim_days'] = (created - uploaded).dt.days
    out['claim_year'] = created.dt.year
    out['claim_month'] = created.dt.month
    out['title_len'] = df['video_title'].fillna('').str.len()

    for col in LOW_CARD + HIGH_CARD:
        out[col] = df[col].fillna('<missing>').astype(str)
    out['_created'] = created
    return out


def make_model():
    prep = ColumnTransformer(
        [
            ('num', 'passthrough', NUMERIC),
            ('low', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), LOW_CARD),
            ('high', TargetEncoder(target_type='binary', smooth='auto', random_state=RANDOM_STATE), HIGH_CARD),
        ],
        verbose_feature_names_out=False,
    )
    clf = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.1,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        class_weight='balanced',
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        categorical_features=list(range(len(NUMERIC), len(NUMERIC) + len(LOW_CARD))),
    )
    return Pipeline([('prep', prep), ('clf', clf)])


def tune_threshold(y_true, proba):
    """Decision threshold maximizing balanced accuracy."""
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        ba = balanced_accuracy_score(y_true, (proba >= t).astype(int))
        if ba > best_ba:
            best_t, best_ba = float(t), float(ba)
    return best_t, best_ba


def tune_triage_cutoffs(y_true, proba):
    """Most permissive cutoffs whose auto buckets stay >= BUCKET_TARGET accurate.

    AUTO_N: proba <= t_low, AUTO_Y: proba >= t_high. Falls back to disabling a
    bucket (t_low=0.0 / t_high=1.01) if the target cannot be met.
    """
    grid = np.round(np.arange(0.0, 1.0001, 0.005), 4)
    t_low, t_high = 0.0, 1.01
    for t in grid:
        mask = proba <= t
        if mask.sum() >= MIN_BUCKET_N and (y_true[mask] == 0).mean() >= BUCKET_TARGET:
            t_low = float(t)
    for t in grid[::-1]:
        mask = proba >= t
        if mask.sum() >= MIN_BUCKET_N and (y_true[mask] == 1).mean() >= BUCKET_TARGET:
            t_high = float(t)
    return t_low, t_high


def train_model(training_data, status_callback=None, check_stopped=None):
    """Fit the classifier and calibrate thresholds; persist to MODEL_PATH."""

    def status(msg):
        if check_stopped:
            check_stopped(msg)
        if status_callback:
            status_callback(msg)

    status('Loading training data')
    df = pandas.read_csv(training_data, low_memory=False)
    df['verdict'] = df['verdict'].astype(str).str.upper()
    df = df[df['verdict'].isin(['Y', 'N'])]
    if 'no_code' in df.columns:
        df = df[~df['no_code'].isin(excluded_codes)]

    X = build_features(df)
    y = (df['verdict'] == 'Y').astype(int).to_numpy()

    # Temporal split: fit on older claims, tune thresholds on the most recent
    # ones (closest to the claims the model will actually score).
    in_window = (X['_created'] >= TRAIN_START).to_numpy()
    holdout_start = X['_created'].max() - pandas.Timedelta(days=HOLDOUT_DAYS)
    val_mask = in_window & (X['_created'] >= holdout_start).to_numpy()
    fit_mask = in_window & ~val_mask
    if val_mask.sum() < 500 or fit_mask.sum() < 5000:
        raise ValueError(
            f'Not enough labeled claims for temporal split '
            f'(fit={fit_mask.sum()}, holdout={val_mask.sum()})')

    status(f'Training model on {fit_mask.sum()} claims '
           f'(holdout {val_mask.sum()} for threshold tuning)')
    model = make_model()
    model.fit(X.loc[fit_mask, FEATURES], y[fit_mask])

    proba_val = model.predict_proba(X.loc[val_mask, FEATURES])[:, 1]
    threshold, val_ba = tune_threshold(y[val_mask], proba_val)
    t_low, t_high = tune_triage_cutoffs(y[val_mask], proba_val)
    val_auc = float(roc_auc_score(y[val_mask], proba_val))
    status(f'Holdout: auc={val_auc:.4f} balanced_acc={val_ba:.4f} '
           f'@ threshold={threshold:.2f}; triage cutoffs=({t_low}, {t_high})')

    status('Refitting on all claims in training window')
    final_model = make_model()
    final_model.fit(X.loc[in_window, FEATURES], y[in_window])

    artifact = {
        'model': final_model,
        'threshold': threshold,
        't_low': t_low,
        't_high': t_high,
        'metadata': {
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'training_data': str(training_data),
            'train_rows': int(in_window.sum()),
            'holdout_rows': int(val_mask.sum()),
            'holdout_auc': val_auc,
            'holdout_balanced_accuracy': val_ba,
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    global _ARTIFACT
    _ARTIFACT = artifact          # a retrain replaces what the process serves
    status(f'Saved model artifact to {MODEL_PATH}')
    return artifact


# ---------------------------------------------------------------------------
# Video availability (lazy YouTube client so offline runs need no API key)
# ---------------------------------------------------------------------------
def get_youtube_client():
    load_env(['YT_API_KEY'])
    from googleapiclient.discovery import build
    return build('youtube', 'v3', developerKey=os.environ['YT_API_KEY'])


def check_videos_available_batch(video_ids, youtube_client, check_stopped=None):
    """
    Check video availability in batches of 50 (API limit)
    Mark entire batch as unavailable on error
    Returns dict mapping video_id -> availability (True/False)
    """
    from tqdm import tqdm

    results = {}
    batch_size = 50
    num_batches = min(int(len(video_ids) / batch_size), 10000) + 1
    msg = 'Checking video availability'

    for i in tqdm(range(0, len(video_ids), batch_size), total=num_batches, desc=msg):
        if check_stopped:
            check_stopped(msg)
        batch = video_ids[i:i + batch_size]

        try:
            request = youtube_client.videos().list(part='id', id=','.join(batch))
            response = request.execute()
            found_ids = {item['id'] for item in response.get('items', [])}

            for video_id in batch:
                results[video_id] = video_id in found_ids

        except Exception:
            for video_id in batch:
                results[video_id] = False

    return results


# ---------------------------------------------------------------------------
# Title language detection (offline fastText lid.176; no API key or network)
# ---------------------------------------------------------------------------
# lid.176 labels are mostly ISO 639-1; translate those to the ISO 639-3 codes
# used by the WESS sheet. Labels not in this dict (ceb, war, arz, ...) are
# already 639-3-compatible and pass through unchanged.
FASTTEXT_TO_ISO3 = {
    'af': 'afr', 'am': 'amh', 'an': 'arg', 'ar': 'ara', 'as': 'asm',
    'av': 'ava', 'az': 'aze', 'ba': 'bak', 'be': 'bel', 'bg': 'bul',
    'bn': 'ben', 'bo': 'bod', 'br': 'bre', 'bs': 'bos', 'ca': 'cat',
    'ce': 'che', 'co': 'cos', 'cs': 'ces', 'cv': 'chv', 'cy': 'cym',
    'da': 'dan', 'de': 'deu', 'dv': 'div', 'el': 'ell', 'en': 'eng',
    'eo': 'epo', 'es': 'spa', 'et': 'est', 'eu': 'eus', 'fa': 'fas',
    'fi': 'fin', 'fr': 'fra', 'fy': 'fry', 'ga': 'gle', 'gd': 'gla',
    'gl': 'glg', 'gn': 'grn', 'gu': 'guj', 'gv': 'glv', 'he': 'heb',
    'hi': 'hin', 'hr': 'hrv', 'ht': 'hat', 'hu': 'hun', 'hy': 'hye',
    'ia': 'ina', 'id': 'ind', 'ie': 'ile', 'io': 'ido', 'is': 'isl',
    'it': 'ita', 'ja': 'jpn', 'jv': 'jav', 'ka': 'kat', 'kk': 'kaz',
    'km': 'khm', 'kn': 'kan', 'ko': 'kor', 'ku': 'kur', 'kv': 'kom',
    'kw': 'cor', 'ky': 'kir', 'la': 'lat', 'lb': 'ltz', 'li': 'lim',
    'lo': 'lao', 'lt': 'lit', 'lv': 'lav', 'mg': 'mlg', 'mk': 'mkd',
    'ml': 'mal', 'mn': 'mon', 'mr': 'mar', 'ms': 'msa', 'mt': 'mlt',
    'my': 'mya', 'ne': 'nep', 'nl': 'nld', 'nn': 'nno', 'no': 'nor',
    'oc': 'oci', 'or': 'ori', 'os': 'oss', 'pa': 'pan', 'pl': 'pol',
    'ps': 'pus', 'pt': 'por', 'qu': 'que', 'rm': 'roh', 'ro': 'ron',
    'ru': 'rus', 'sa': 'san', 'sc': 'srd', 'sd': 'snd', 'sh': 'hbs',
    'si': 'sin', 'sk': 'slk', 'sl': 'slv', 'so': 'som', 'sq': 'sqi',
    'sr': 'srp', 'su': 'sun', 'sv': 'swe', 'sw': 'swa', 'ta': 'tam',
    'te': 'tel', 'tg': 'tgk', 'th': 'tha', 'tk': 'tuk', 'tl': 'tgl',
    'tr': 'tur', 'tt': 'tat', 'ug': 'uig', 'uk': 'ukr', 'ur': 'urd',
    'uz': 'uzb', 'vi': 'vie', 'vo': 'vol', 'wa': 'wln', 'yi': 'yid',
    'yo': 'yor', 'zh': 'zho',
    'als': 'gsw',  # fastText 'als' is Alemannic, not ISO 639-3 Tosk Albanian
    'bh': 'bih',   # Bihari collective
}

# Individual language -> ISO 639-3 macrolanguage, so a WESS individual code
# (e.g. zlm) matches a detection lid can only make at macro level (ms -> msa).
# Members limited to codes present in the WESS sheet or the lid label set.
# Serbo-Croatian is deliberately not collapsed: lid distinguishes sr/hr/bs.
MACRO_OF = {
    # Malay (lid: ms, id, min)
    'zlm': 'msa', 'zsm': 'msa', 'ind': 'msa', 'min': 'msa', 'bjn': 'msa',
    'jax': 'msa', 'kvr': 'msa', 'liw': 'msa', 'max': 'msa', 'mfa': 'msa',
    'mfb': 'msa', 'mqg': 'msa', 'mui': 'msa', 'pse': 'msa', 'vkt': 'msa',
    'xmm': 'msa',
    # Arabic (lid: ar, arz)
    'arb': 'ara', 'arz': 'ara', 'acm': 'ara', 'acq': 'ara', 'aeb': 'ara',
    'aec': 'ara', 'afb': 'ara', 'ajp': 'ara', 'apd': 'ara', 'arq': 'ara',
    'ary': 'ara', 'avl': 'ara', 'ayl': 'ara', 'shu': 'ara',
    # Chinese (lid: zh, yue, wuu)
    'cmn': 'zho', 'yue': 'zho', 'wuu': 'zho', 'nan': 'zho', 'hak': 'zho',
    'cdo': 'zho', 'hsn': 'zho',
    'pes': 'fas', 'prs': 'fas',              # Persian (lid: fa)
    'swh': 'swa', 'swc': 'swa',              # Swahili (lid: sw)
    'azj': 'aze', 'azb': 'aze',              # Azerbaijani (lid: az, azb)
    'uzn': 'uzb', 'uzs': 'uzb',              # Uzbek (lid: uz)
    'npi': 'nep', 'dty': 'nep',              # Nepali (lid: ne, dty)
    'ory': 'ori', 'spv': 'ori',              # Odia (lid: or)
    'kmr': 'kur', 'ckb': 'kur', 'sdh': 'kur',  # Kurdish (lid: ku, ckb)
    'khk': 'mon', 'mvf': 'mon',              # Mongolian (lid: mn)
    'gaz': 'orm', 'hae': 'orm', 'gax': 'orm',  # Oromo
    'gug': 'grn', 'gui': 'grn', 'gun': 'grn', 'gnw': 'grn',  # Guarani (lid: gn)
    'ayr': 'aym',                            # Aymara
    'pbt': 'pus', 'pbu': 'pus',              # Pashto (lid: ps)
    'pnb': 'lah', 'skr': 'lah', 'hnd': 'lah', 'hno': 'lah',  # Lahnda (lid: pnb)
    'nob': 'nor', 'nno': 'nor',              # Norwegian (lid: no, nn)
    'ekk': 'est',                            # Estonian (lid: et)
    'lvs': 'lav',                            # Latvian (lid: lv)
    'src': 'srd',                            # Sardinian (lid: sc)
    'als': 'sqi', 'aln': 'sqi',              # Albanian (lid: sq)
    'gom': 'kok', 'knn': 'kok',              # Konkani (lid: gom)
    'ydd': 'yid',                            # Yiddish (lid: yi)
    # Malagasy (lid: mg)
    'bhr': 'mlg', 'msh': 'mlg', 'plt': 'mlg', 'skg': 'mlg', 'tdx': 'mlg',
    'txy': 'mlg', 'xmv': 'mlg', 'xmw': 'mlg',
    # Quechua (lid: qu)
    'qub': 'que', 'quf': 'que', 'qug': 'que', 'quh': 'que', 'qul': 'que',
    'qup': 'que', 'quy': 'que', 'quz': 'que', 'qva': 'que', 'qve': 'que',
    'qvh': 'que', 'qvi': 'que', 'qvm': 'que', 'qvn': 'que', 'qvs': 'que',
    'qvw': 'que', 'qwh': 'que', 'qxh': 'que', 'qxn': 'que', 'qxo': 'que',
    'qxq': 'que', 'qxr': 'que',
    # Zhuang
    'zch': 'zha', 'zhn': 'zha', 'zyb': 'zha', 'zyj': 'zha',
}


def normalize_fasttext_label(label):
    """'__label__en' -> 'eng'; 3-letter lid labels pass through."""
    code = label.replace('__label__', '').lower()
    return FASTTEXT_TO_ISO3.get(code, code)


# Generic modifiers and ambiguous English words that would cause false name
# matches ("Ave Maria", "Black ..."); tokens < 4 chars are dropped by rule,
# except distinctive short language names in NAME_SHORT_OK.
NAME_STOP_TOKENS = frozenset({
    'wider', 'central', 'south', 'north', 'east', 'west',
    'southern', 'northern', 'eastern', 'western',
    'northeast', 'northwest', 'southeast', 'southwest',
    'upper', 'lower', 'inner', 'outer',
    'global', 'formal', 'general', 'standard', 'modern', 'classical',
    'vehicular', 'emigre', 'language',
    'black', 'maria', 'male', 'mango', 'bench',
})
NAME_SHORT_OK = frozenset({'twi', 'ewe', 'fon', 'lao', 'edo', 'tiv', 'vai'})


def build_lang_name_patterns(lang_df):
    """
    Macro ISO 639-3 -> compiled regex matching any distinctive token of the
    group's anglicized names ('Castilian Spanish' -> castilian|spanish), so a
    title that NAMES its expected language ('JESUS Film - Amharic') can be
    recognized even though it detects as eng.
    """
    tokens = {}
    iso = lang_df['ISO_lang'].fillna('').astype(str).str.strip().str.lower()
    names = lang_df['Anglicized_name'].fillna('').astype(str)
    for code, name in zip(iso, names):
        if len(code) != 3:
            continue
        macro = MACRO_OF.get(code, code)
        for tok in re.split(r'[,\s]+', name.strip().lower()):
            if tok in NAME_STOP_TOKENS:
                continue
            if len(tok) >= 4 or tok in NAME_SHORT_OK:
                tokens.setdefault(macro, set()).add(re.escape(tok))
    return {m: re.compile(r'\b(?:%s)\b' % '|'.join(sorted(t)), re.IGNORECASE)
            for m, t in tokens.items()}


def get_lid_model():
    import fasttext  # provided by fasttext-predict (prediction-only wheels)
    return fasttext.load_model(str(LID_MODEL_PATH))


def detect_languages_batch(titles, check_stopped=None):
    """
    Detect the language of each title with the bundled fastText lid.176 model.
    Labels are normalized to ISO 639-3; predictions below LANG_MIN_PROB map
    to '' (undetermined). Returns dict mapping title -> language code.
    """
    from tqdm import tqdm

    model = get_lid_model()
    results = {}
    msg = 'Detecting title languages'
    for i, title in enumerate(tqdm(titles, desc=msg)):
        if check_stopped and i % 1000 == 0:
            check_stopped(msg)
        labels, probs = model.predict(title.replace('\n', ' '), k=1)
        if labels and probs[0] >= LANG_MIN_PROB:
            results[title] = normalize_fasttext_label(labels[0])
        else:
            results[title] = ''
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(args, status_callback=None, stop_check=None):

    def check_stopped(stage=''):
        if stop_check and stop_check():
            message = f"Task stopped during {stage if stage else 'by user'}"
            raise TaskStoppedError(message)

    # -----------------------------------------------------------------------
    # Model: load cached artifact, or train from the provided export.
    # Training always tunes the decision threshold and triage cutoffs on a
    # temporal holdout (--skip-validation is accepted for compatibility).
    # -----------------------------------------------------------------------
    if MODEL_PATH.exists():
        if status_callback:
            status_callback(f'Loading cached model from {MODEL_PATH}')
        artifact = get_artifact()
    else:
        if not args.training_data:
            raise ValueError(f'Training data is required to create {MODEL_PATH}')
        artifact = train_model(args.training_data,
                               status_callback=status_callback,
                               check_stopped=check_stopped)

    model = artifact['model']
    threshold = artifact['threshold']
    t_low, t_high = artifact['t_low'], artifact['t_high']

    # -----------------------------------------------------------------------
    # Load licensed assets and asset->media_component mapping
    # -----------------------------------------------------------------------
    licensed_df = pandas.read_csv(LICENSED_PATH)
    licensed_asset_ids = set(licensed_df['asset_id'].dropna().unique())

    assets_media_df = pandas.read_csv(ASSET_MEDIA_PATH)
    asset_to_media_component = dict(zip(assets_media_df['asset_id'],
                                        assets_media_df['media_component_id']))

    # WESS language number -> ISO 639-3 (rows with blank ISO codes dropped)
    lang_df = pandas.read_csv(LANG_SHEET_PATH)
    iso = lang_df['ISO_lang'].fillna('').astype(str).str.strip().str.lower()
    nums = pandas.to_numeric(lang_df['WESS_LAN_num'], errors='coerce')
    wess_to_iso = {int(n): c for n, c in zip(nums, iso)
                   if pandas.notna(n) and len(c) == 3}
    lang_name_pats = build_lang_name_patterns(lang_df)

    # -----------------------------------------------------------------------
    # Process unprocessed claims
    # -----------------------------------------------------------------------
    df = pandas.read_csv(args.prediction_input, low_memory=False)

    # Add licensed boolean column
    df['licensed'] = df['asset_id'].isin(licensed_asset_ids)

    # Add media_component_id column
    df['media_component_id'] = df['asset_id'].map(asset_to_media_component)

    # Add video availability column (True if available, False if blocked or
    # unavailable). If the input already carries a video_available column, it
    # is reused and the YouTube API is not called.
    if 'video_available' in df.columns:
        df['video_available'] = (
            df['video_available'].map({True: True, False: False,
                                       'True': True, 'False': False})
            .fillna(True).astype(bool))
    elif 'video_id' in df.columns:
        msg = 'Checking video availability'
        check_stopped(msg)
        if status_callback:
            status_callback(msg)
        youtube = get_youtube_client()
        available_map = check_videos_available_batch(
            df['video_id'].tolist(), youtube, check_stopped=check_stopped)
        df['video_available'] = df['video_id'].map(available_map)
    else:
        df['video_available'] = True  # Default to True if no video_id column

    # predicted_lang: ISO 639-3 of video_title via bundled fastText lid.176
    # ('' = empty title or below-confidence detection). A pre-existing
    # predicted_lang column in the input is reused without re-detecting.
    if 'predicted_lang' in df.columns:
        df['predicted_lang'] = df['predicted_lang'].fillna('').astype(str)
    elif 'video_title' in df.columns:
        msg = 'Detecting title languages'
        check_stopped(msg)
        if status_callback:
            status_callback(msg)
        stripped = df['video_title'].fillna('').astype(str).str.strip()
        unique_titles = [t for t in stripped.unique() if t]
        lang_map = detect_languages_batch(unique_titles,
                                          check_stopped=check_stopped)
        df['predicted_lang'] = stripped.map(lang_map).fillna('')
    else:
        df['predicted_lang'] = ''

    # expected_lang: claim's WESS language_id mapped to ISO 639-3 via
    # sheets_language_families.csv ('' when blank or unmapped).
    if 'language_id' in df.columns:
        wess_num = pandas.to_numeric(df['language_id'],
                                     errors='coerce').astype('Int64')
        df['expected_lang'] = wess_num.map(wess_to_iso).fillna('')
    else:
        df['expected_lang'] = ''

    # lang_match: Y/N with macrolanguage-aware equality (zlm vs ms -> Y),
    # '' when either side is undetermined. Titles that NAME the expected
    # language ('JESUS Film - Amharic' detects as eng) are not mismatches:
    # title_names_lang records the name hit and upgrades N -> Y.
    pred = df['predicted_lang'].map(lambda c: MACRO_OF.get(c, c))
    exp = df['expected_lang'].map(lambda c: MACRO_OF.get(c, c))
    titles = (df['video_title'].fillna('').astype(str)
              if 'video_title' in df.columns
              else pandas.Series('', index=df.index))
    named = pandas.Series(False, index=df.index)
    for code, sub in titles.groupby(exp):
        if code in lang_name_pats:
            named.loc[sub.index] = sub.str.contains(lang_name_pats[code])
    df['title_names_lang'] = np.where(
        (titles == '') | ~exp.isin(list(lang_name_pats)), '',
        np.where(named, 'Y', 'N'))
    df['lang_match'] = np.select(
        [(pred == '') | (exp == ''), pred == exp, named],
        ['', 'Y', 'Y'], 'N')

    check_stopped('before predictions')
    if status_callback:
        status_callback('Making predictions')

    # -----------------------------------------------------------------------
    # Predict + rules + triage
    # -----------------------------------------------------------------------
    features = build_features(df)
    proba = model.predict_proba(features[FEATURES])[:, 1]

    licensed = df['licensed'].to_numpy(dtype=bool)
    unavailable = ~df['video_available'].to_numpy(dtype=bool)
    forced_n = licensed | unavailable

    # rating stays the model probability, zeroed by rules (existing contract)
    df['rating'] = np.where(forced_n, 0.0, proba)
    df['predicted_verdict'] = np.where(~forced_n & (proba >= threshold), 'Y', 'N')
    df['confidence'] = np.maximum(proba, 1 - proba).round(4)
    df['triage'] = np.select(
        [licensed, unavailable, proba <= t_low, proba >= t_high],
        ['AUTO_N_LICENSED', 'AUTO_N_UNAVAILABLE', 'AUTO_N', 'AUTO_Y'],
        default='REVIEW')

    # Save predictions
    df.to_csv(args.prediction_output, index=False)


class TaskStoppedError(Exception):
    """Custom exception to indicate task was stopped."""
    pass


if __name__ == '__main__':

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process claims data and train classifier')
    parser.add_argument('--training-data', default=None,
                        help='Training data CSV (required when model.joblib is absent)')
    parser.add_argument('--prediction-input', required=True, help='Input CSV for prediction')
    parser.add_argument('--prediction-output', default=f'ml_enriched_claims_{datetime.now().strftime("%Y%m%d%H%M")}.csv', help='Output CSV')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Accepted for compatibility; validation/threshold tuning always runs at training time')

    args = parser.parse_args()

    main(args, status_callback=print)

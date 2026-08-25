import argparse
import os
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

RANDOM_STATE = 42
TRAIN_START = pandas.Timestamp('2022-01-01')  # recency window that validated best
HOLDOUT_DAYS = 150          # most recent labeled claims, used to tune thresholds
BUCKET_TARGET = 0.95        # per-bucket accuracy target for triage cutoffs
MIN_BUCKET_N = 50

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
# Title language detection (lazy key so runs without it just skip the column)
# ---------------------------------------------------------------------------
def get_google_api_key():
    load_env()
    return os.environ.get('GOOGLE_API_KEY')


def detect_languages_batch(titles, api_key, check_stopped=None):
    """
    Detect the language of each title via Google Translation v2 detect,
    in batches of 128 (API limit), retrying with backoff on errors.
    Chunks that still fail map to '' (empty string).
    Returns dict mapping title -> language code.
    """
    import time

    import requests
    from tqdm import tqdm

    url = 'https://translation.googleapis.com/language/translate/v2/detect'
    results = {}
    batch_size = 128
    max_retries = 3
    msg = 'Detecting title languages'

    for i in tqdm(range(0, len(titles), batch_size), desc=msg):
        if check_stopped:
            check_stopped(msg)
        batch = titles[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                response = requests.post(url, params={'key': api_key},
                                         json={'q': batch}, timeout=30)
                response.raise_for_status()
                detections = response.json()['data']['detections']
                for title, detection in zip(batch, detections):
                    results[title] = detection[0]['language'] if detection else ''
                break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            for title in batch:
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
        artifact = joblib.load(MODEL_PATH)
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

    # Add predicted_lang column (language code of video_title via Google
    # Translation detect; '' = not attempted/failed, 'und' = undetermined).
    # If the input already carries a predicted_lang column, it is reused and
    # the Translation API is not called. Missing GOOGLE_API_KEY skips detection.
    if 'predicted_lang' in df.columns:
        df['predicted_lang'] = df['predicted_lang'].fillna('').astype(str)
    elif 'video_title' in df.columns:
        api_key = get_google_api_key()
        if api_key:
            msg = 'Detecting title languages'
            check_stopped(msg)
            if status_callback:
                status_callback(msg)
            stripped = df['video_title'].fillna('').astype(str).str.strip()
            unique_titles = [t for t in stripped.unique() if t]
            lang_map = detect_languages_batch(unique_titles, api_key,
                                              check_stopped=check_stopped)
            df['predicted_lang'] = stripped.map(lang_map).fillna('')
        else:
            if status_callback:
                status_callback('GOOGLE_API_KEY not set; skipping language detection')
            df['predicted_lang'] = ''
    else:
        df['predicted_lang'] = ''

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

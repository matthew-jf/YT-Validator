import argparse
import json
import os
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
CHANNEL_VERDICTS_PATH = BASE_DIR / 'channel_verdicts.csv'

RANDOM_STATE = 42
TRAIN_START = pandas.Timestamp('2022-01-01')  # recency window that validated best
HOLDOUT_DAYS = 150          # most recent labeled claims, used to tune thresholds
BUCKET_TARGET = 0.95        # per-bucket accuracy target for triage cutoffs
MIN_BUCKET_N = 50
MIN_CHANNEL_CLAIMS = 3      # labeled claims a unanimous channel needs before its
                            # history can upgrade REVIEW rows to AUTO_N_CHANNEL
CHANNEL_RATING_CAP = 0.025  # ...and the model probability must also be this low.
                            # Calibrated on the July 2026 reviewed batch: 95.4%
                            # accurate there; without the cap the bucket is 83%.

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

# What videos.list can tell us about a video. 'missing' covers deleted and
# private alike: the API returns neither, and cannot say which.
VIDEO_STATES = ('public', 'unlisted', 'missing')

NUMERIC = [
    'views_log', 'matching_duration', 'longest_match', 'video_duration_sec',
    'match_ratio', 'longest_ratio', 'match_per_longest',
    'upload_to_claim_days', 'claim_year', 'claim_month', 'title_len',
]
LOW_CARD = ['claim_type', 'claim_origin', 'content_type', 'claim_report_source', 'asset_labels']
HIGH_CARD = ['asset_id', 'custom_id', 'reference_id', 'channel_id']
FEATURES = NUMERIC + LOW_CARD + HIGH_CARD


def read_claims_csv(path):
    """Read a claims export, tolerating the malformed rows some exports contain.

    Unbalanced quotes crash pandas' C parser with "Buffer overflow caught";
    fall back to the slower python parser and drop the handful of bad lines.
    """
    try:
        return pandas.read_csv(path, low_memory=False)
    except pandas.errors.ParserError:
        return pandas.read_csv(path, engine='python', on_bad_lines='skip')


def build_channel_verdicts(labeled_df, out_path=CHANNEL_VERDICTS_PATH):
    """Derive each channel's unanimous verdict history from labeled claims.

    Keeps only channels whose Y/N verdicts all agree. Rule-driven verdicts
    (no_code L/V/N/X) are dropped when the export carries no_code, mirroring
    the training exclusions. Writes a bundled CSV (channel_id, verdict,
    n_claims) that main() consumes at predict time.
    """
    df = labeled_df.assign(verdict=labeled_df['verdict'].astype(str).str.upper())
    df = df[df['verdict'].isin(['Y', 'N'])]
    if 'no_code' in df.columns:
        df = df[~df['no_code'].astype(str).str.upper().isin(excluded_codes)]
    df = df.dropna(subset=['channel_id'])

    per_channel = df.groupby('channel_id')['verdict'].agg(['first', 'nunique', 'size'])
    unanimous = per_channel[per_channel['nunique'] == 1]
    out = pandas.DataFrame({
        'channel_id': unanimous.index,
        'verdict': unanimous['first'].to_numpy(),
        'n_claims': unanimous['size'].to_numpy(),
    })
    out.to_csv(out_path, index=False)
    return out


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
    df = read_claims_csv(training_data)
    df['verdict'] = df['verdict'].astype(str).str.upper()
    df = df[df['verdict'].isin(['Y', 'N'])]
    if 'no_code' in df.columns:
        df = df[~df['no_code'].isin(excluded_codes)]

    status('Rebuilding channel verdict history')
    build_channel_verdicts(df)

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


def normalize_video_id(value):
    """Strip the apostrophe Excel adds to ids that begin with '-'."""
    return str(value).strip().lstrip("'")


# Reasons no amount of retrying will fix: the key is spent, wrong, or barred.
# Everything else (a 5xx, a rate limit, a dropped connection) is worth retrying.
FATAL_API_REASONS = {'quotaExceeded', 'quotaExceeded402', 'dailyLimitExceeded',
                     'dailyLimitExceeded402', 'dailyLimitExceededUnreg',
                     'variableTermLimitExceeded', 'variableTermExpiredDailyExceeded',
                     'keyInvalid', 'keyExpired', 'accessNotConfigured', 'forbidden'}


class AvailabilityLookupStopped(Exception):
    """The API refused in a way that retrying cannot fix. Carries the reason."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _api_error_reason(exc):
    """The API's reason code ('quotaExceeded', ...) from an HttpError.

    HttpError.reason is the human-readable message, not the code, so the code
    comes from the response body.
    """
    content = getattr(exc, 'content', None)
    if content is None:
        return ''
    try:
        error = json.loads(content.decode('utf-8'))['error']
    except (ValueError, KeyError, TypeError, AttributeError):
        return ''
    for item in list(error.get('errors') or []) + list(error.get('details') or []):
        if isinstance(item, dict) and item.get('reason'):
            return item['reason']
    return ''


def _lookup_ids(youtube_client, ids, retries):
    """What YouTube says about these ids: id -> privacyStatus. None means the
    call kept failing.

    `status` costs nothing extra -- videos.list bills 1 unit per call whatever
    the parts -- and it is the difference between "this video is gone" and
    "this video is unlisted", which the id alone cannot tell apart.
    """
    import time
    for attempt in range(retries + 1):
        try:
            response = youtube_client.videos().list(part='id,status',
                                                    id=','.join(ids)).execute()
            return {item['id']: (item.get('status') or {}).get('privacyStatus') or 'public'
                    for item in response.get('items', [])}
        except Exception as exc:
            reason = _api_error_reason(exc)
            if reason in FATAL_API_REASONS:
                # Quota does not come back in two seconds, and a bad key never
                # does. Stop the whole lookup rather than spend ~800 doomed
                # calls discovering it one batch at a time.
                raise AvailabilityLookupStopped(reason) from exc
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                print(f'Availability lookup failed for {len(ids)} id(s): {exc}')
    return None


OUTAGE_GIVE_UP = 3   # consecutive single-id failures that mean "not one bad id"


def check_videos_available_batch(video_ids, youtube_client, check_stopped=None,
                                 retries=2):
    """
    What state each video is in, in batches of 50 (the API limit).

    Returns video_id -> 'public' / 'unlisted' / 'missing' / None (unknown).
    Callers must treat None as unknown, never as missing: a failed lookup is
    not evidence about the video.

    'missing' means the API did not return the video, which is deleted and
    private alike -- and sometimes neither, since a partial response omits
    videos that are perfectly fine. Ids absent from a batch are therefore
    asked for a second time, alone, before being called missing. On a sample
    of 183 ids this flag had called 44 of them gone when they were public.
    """
    from tqdm import tqdm

    results = {}
    stopped_reason = None
    batch_size = 50
    num_batches = min(int(len(video_ids) / batch_size), 10000) + 1
    msg = 'Checking video availability'

    for i in tqdm(range(0, len(video_ids), batch_size), total=num_batches, desc=msg):
        if check_stopped:
            check_stopped(msg)
        if stopped_reason:
            break
        batch = video_ids[i:i + batch_size]
        lookup = {normalize_video_id(v): v for v in batch}

        try:
            found = _lookup_ids(youtube_client, list(lookup), retries)
        except AvailabilityLookupStopped as stop:
            stopped_reason = stop.reason
            break
        if found is not None:
            # Confirm an absence before trusting it: one truncated response
            # would otherwise mark a live video gone for good. The second ask
            # goes as one more batch, not one call per id -- a quarter of a
            # monthly batch is genuinely missing, and asking singly would cost
            # 50 units a head.
            absent = [clean for clean in lookup if clean not in found]
            if absent:
                again = _lookup_ids(youtube_client, absent, retries)
                if again:
                    found = {**found, **again}
            for clean, original in lookup.items():
                results[original] = found.get(clean, 'missing')
            continue

        # The batch call kept failing. Retry the ids one at a time so a single
        # unacceptable id cannot condemn the other 49 to review; give up once
        # several in a row fail, which means the API is down rather than an id
        # being bad, and leave the rest unknown.
        consecutive_failures = 0
        for clean, original in lookup.items():
            if consecutive_failures >= OUTAGE_GIVE_UP:
                results[original] = None
                continue
            try:
                one = _lookup_ids(youtube_client, [clean], retries=0)
            except AvailabilityLookupStopped as stop:
                stopped_reason = stop.reason
                break
            if one is None:
                consecutive_failures += 1
                results[original] = None
            else:
                consecutive_failures = 0
                results[original] = one.get(clean, 'missing')

    # Everything not reached stays out of `results`, so the caller reads it as
    # unknown and routes it to a person -- never as a missing video.
    return results, stopped_reason


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
    # Load licensed assets, asset->media_component mapping, channel history
    # -----------------------------------------------------------------------
    licensed_df = pandas.read_csv(LICENSED_PATH)
    licensed_asset_ids = set(licensed_df['asset_id'].dropna().unique())

    assets_media_df = pandas.read_csv(ASSET_MEDIA_PATH)
    asset_to_media_component = dict(zip(assets_media_df['asset_id'],
                                        assets_media_df['media_component_id']))

    channel_hist = pandas.read_csv(CHANNEL_VERDICTS_PATH).set_index('channel_id')

    # -----------------------------------------------------------------------
    # Process unprocessed claims
    # -----------------------------------------------------------------------
    df = read_claims_csv(args.prediction_input)

    # Add licensed boolean column
    df['licensed'] = df['asset_id'].isin(licensed_asset_ids)

    # Add media_component_id column
    df['media_component_id'] = df['asset_id'].map(asset_to_media_component)

    # What state each video is in: 'public', 'unlisted', 'missing' or NaN for
    # unknown. An input that already carries video_state is reused and the
    # YouTube API is not called; an older input carrying only the video_available
    # boolean is read as the two states it could express.
    if 'video_state' in df.columns:
        df['video_state'] = df['video_state'].where(
            df['video_state'].isin(VIDEO_STATES))
    elif 'video_available' in df.columns:
        # NaN stays NaN: an unknown state must not become 'public' any more
        # than it may become 'missing'.
        df['video_state'] = df['video_available'].map(
            {True: 'public', False: 'missing', 'True': 'public', 'False': 'missing'})
    elif 'video_id' in df.columns:
        msg = 'Checking video availability'
        check_stopped(msg)
        if status_callback:
            status_callback(msg)
        youtube = get_youtube_client()
        state_map, stopped_reason = check_videos_available_batch(
            df['video_id'].tolist(), youtube, check_stopped=check_stopped)
        df['video_state'] = df['video_id'].map(state_map)
        if stopped_reason:
            # Say so loudly and in the file: a batch where every row is unknown
            # and goes to review otherwise looks like a model that gave up.
            unchecked = int(df['video_state'].isna().sum())
            note = (f'availability lookup stopped: {stopped_reason} '
                    f'({unchecked} of {len(df)} rows unchecked)')
            df.loc[df['video_state'].isna(), 'video_state_note'] = note
            print(note)
            if status_callback:
                status_callback(note)
    else:
        df['video_state'] = 'public'  # no video_id column: nothing to look up

    # Kept for every consumer that still reads the boolean. Unlisted counts as
    # available: it is watchable by link and earns views like any other video.
    df['video_available'] = df['video_state'].map(
        {'public': True, 'unlisted': True, 'missing': False})

    check_stopped('before predictions')
    if status_callback:
        status_callback('Making predictions')

    # -----------------------------------------------------------------------
    # Predict + rules + triage
    # -----------------------------------------------------------------------
    features = build_features(df)
    proba = model.predict_proba(features[FEATURES])[:, 1]

    licensed = df['licensed'].to_numpy(dtype=bool)
    # Availability no longer rejects anything. Graded against a reviewed batch,
    # auto-rejecting a missing video was right 88% of the time -- Ben still
    # calls 12% of gone videos a valid claim, because he judges the claim, not
    # whether the video is up today. So the state routes to a person and is
    # shown alongside the row; it never decides.
    unavailable = (df['video_state'] == 'missing').to_numpy()
    unknown_availability = df['video_state'].isna().to_numpy()
    forced_n = licensed

    # rating stays the model probability, zeroed by rules (existing contract)
    df['rating'] = np.where(forced_n, 0.0, proba)
    df['predicted_verdict'] = np.where(~forced_n & (proba >= threshold), 'Y', 'N')
    df['confidence'] = np.maximum(proba, 1 - proba).round(4)
    df['triage'] = np.select(
        [licensed, unavailable, unknown_availability,   # both route to REVIEW now
         proba <= t_low, proba >= t_high],
        ['AUTO_N_LICENSED', 'REVIEW', 'REVIEW', 'AUTO_N', 'AUTO_Y'],
        default='REVIEW')

    # Channel-history triage assist. A unanimous channel history alone is too
    # weak to auto-decide (~83% accurate on the July 2026 reviewed batch), so
    # REVIEW rows are upgraded only when the model also strongly leans N
    # (rating <= CHANNEL_RATING_CAP). The channel condition is load-bearing:
    # low-rating REVIEW rows without it are ~88% N, with it ~95%. Unanimous-Y
    # history is exposed for reviewers but never auto-decides (<70% accurate).
    df['channel_history_verdict'] = df['channel_id'].map(channel_hist['verdict'])
    df['channel_history_claims'] = (
        df['channel_id'].map(channel_hist['n_claims']).fillna(0).astype(int))
    channel_auto_n = (
        (df['triage'] == 'REVIEW')
        & (df['predicted_verdict'] == 'N')
        & (df['rating'] <= CHANNEL_RATING_CAP)
        & (df['channel_history_verdict'] == 'N')
        & (df['channel_history_claims'] >= MIN_CHANNEL_CLAIMS)
        # A row sent to REVIEW because the video is missing or its state is
        # unknown stays there: this override would otherwise auto-reject it
        # through the back door.
        & ~(unavailable | unknown_availability)
    )
    df.loc[channel_auto_n, 'triage'] = 'AUTO_N_CHANNEL'

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

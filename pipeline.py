import argparse
from datetime import datetime
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import pandas
import numpy as np
from tqdm import tqdm
from googleapiclient.discovery import build
from autogluon.tabular import TabularPredictor
from helpers import load_env

load_env(["YT_API_KEY"])

# For checking YouTube video availability
youtube = build('youtube', 'v3', developerKey=os.environ["YT_API_KEY"])

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Trey's pipeline: pretrained AutoGluon challenger stack (primary decision
# maker) + calibrated Human-Review threshold. See trey_pipeline/.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get('MODEL_DIR', REPO_ROOT / 'trey_pipeline' / 'models'))

# Which artifact under MODEL_DIR to serve. 'ag_challenger' is the full training
# output; 'ag_challenger_deploy' is the pruned inference-only clone produced by
# trey_pipeline/ml_pipeline/export_deploy.py (same predictions, 55% smaller).
MODEL_NAME = os.environ.get('MODEL_NAME', 'ag_challenger')

sys.path.insert(0, str(REPO_ROOT / 'trey_pipeline' / 'ml_pipeline'))
from feature_utils import engineer_features, AG_FEATURES

AUTO_YES_THRESHOLD = 0.97

# Columns engineer_features / the model need, beyond asset_id used for
# the licensed & media-component lookups.
REQUIRED_INPUT_COLUMNS = [
    'asset_id', 'video_title', 'asset_title', 'video_duration_sec',
    'duration_seconds', 'matching_duration', 'longest_match',
]

# The artifact is ~1.4 GB on disk and takes seconds to deserialise, so it is
# loaded once per process rather than once per request. app.py warms this at
# startup so a broken artifact fails the service immediately instead of
# surfacing inside a background task.
_PREDICTOR = None
_REVIEW_THRESHOLD = None
_MODEL_LOAD_SECONDS = None
_WARM_SECONDS = None
_LOAD_LOCK = threading.Lock()


class TaskStoppedError(Exception):
    """Custom exception to indicate task was stopped."""
    pass


def peak_rss_mb():
    """Process high-water memory mark, in MB. ru_maxrss is bytes on macOS, KB on Linux."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1048576 if sys.platform == 'darwin' else 1024), 1)
    except Exception:
        return None


def get_predictor():
    """Return (predictor, review_threshold), loading once per process."""
    global _PREDICTOR, _REVIEW_THRESHOLD, _MODEL_LOAD_SECONDS
    if _PREDICTOR is None:
        with _LOAD_LOCK:
            if _PREDICTOR is None:  # another thread may have won the race
                started = time.time()
                predictor = TabularPredictor.load(str(MODEL_DIR / MODEL_NAME))
                with open(MODEL_DIR / 'ag_threshold.json') as f:
                    _REVIEW_THRESHOLD = json.load(f)['threshold']
                _MODEL_LOAD_SECONDS = time.time() - started
                _PREDICTOR = predictor
    return _PREDICTOR, _REVIEW_THRESHOLD


def warm_predictor():
    """Force the model files off disk and into memory.

    TabularPredictor.load() only reads predictor/learner metadata; AutoGluon
    defers loading the model files themselves until the first predict. So a
    plain load neither pays the real cost nor detects a corrupt model file.
    Scoring one synthetic row forces both.
    """
    global _WARM_SECONDS
    predictor, _ = get_predictor()
    started = time.time()
    probe = pandas.DataFrame([{
        'duration_diff_sec': 0.0, 'duration_ratio': 1.0,
        'title_fuzzy_ratio': 100, 'title_token_sort_ratio': 100,
        'title_token_set_ratio': 100, 'matching_duration': 60.0,
        'longest_match': 60.0, 'video_title': 'warmup', 'asset_title': 'warmup',
    }])
    predictor.predict_proba(probe[AG_FEATURES])
    _WARM_SECONDS = time.time() - started
    return _WARM_SECONDS


def model_info():
    """Describe the loaded model, for /health and telemetry."""
    return {
        'model_dir': str(MODEL_DIR / MODEL_NAME),
        'loaded': _PREDICTOR is not None,
        'model_best': getattr(_PREDICTOR, 'model_best', None),
        'review_threshold': _REVIEW_THRESHOLD,
        'auto_yes_threshold': AUTO_YES_THRESHOLD,
        'load_seconds': round(_MODEL_LOAD_SECONDS, 2) if _MODEL_LOAD_SECONDS else None,
        'warm_seconds': round(_WARM_SECONDS, 2) if _WARM_SECONDS else None,
        'warm': _WARM_SECONDS is not None,
        'peak_rss_mb': peak_rss_mb(),
    }


# Function to check if YouTube videos are available
def check_videos_available_batch(video_ids, youtube_client, check_stopped=None):
    """
    Check video availability in batches of 50 (API limit)
    Mark entire batch as unavailable on error
    Returns dict mapping video_id -> availability (True/False)
    """
    results = {}
    batch_size = 50
    num_batches = min(int(len(video_ids) / batch_size), 10000) + 1
    msg = "Checking video availability"

    for i in tqdm(range(0, len(video_ids), batch_size), total=num_batches, desc=msg):
        if check_stopped: check_stopped(msg)
        batch = video_ids[i:i + batch_size]

        try:
            request = youtube_client.videos().list(part="id", id=','.join(batch))
            response = request.execute()
            found_ids = {item['id'] for item in response.get('items', [])}

            for video_id in batch:
                results[video_id] = video_id in found_ids

        except Exception as e:
            for video_id in batch:
                results[video_id] = False

    return results


def main(args, status_callback=None, stop_check=None):

    def check_stopped(stage=""):
        if stop_check and stop_check():
            message = f"Task stopped during {stage if stage else 'by user'}"
            raise TaskStoppedError(message)

    telemetry = {'rows': 0}
    run_started = time.time()

    # -----------------------------------------------------------------------
    # Load pretrained model artifacts (cached after the first call)
    # -----------------------------------------------------------------------
    msg = "Loading AutoGluon challenger model"
    check_stopped(msg)
    if status_callback:
        status_callback(msg)

    predictor, review_threshold = get_predictor()
    telemetry['model_load_seconds'] = _MODEL_LOAD_SECONDS

    # -----------------------------------------------------------------------
    # Load licensed assets and asset->media_component mapping
    # -----------------------------------------------------------------------
    licensed_df = pandas.read_csv(REPO_ROOT / 'Licensed.csv')
    licensed_asset_ids = set(licensed_df['asset_id'].dropna().unique())

    assets_media_df = pandas.read_csv(REPO_ROOT / 'assets_single_media_component.csv')
    asset_to_media_component = dict(zip(assets_media_df['asset_id'],
                                        assets_media_df['media_component_id']))

    # -----------------------------------------------------------------------
    # Process unprocessed claims
    # -----------------------------------------------------------------------
    msg = "Loading prediction input"
    check_stopped(msg)
    if status_callback:
        status_callback(msg)

    df = pandas.read_csv(args.prediction_input, engine='python',
                         on_bad_lines='skip', encoding='utf-8-sig')
    df.columns = df.columns.astype(str).str.strip()

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required columns: {missing}")

    # Add licensed boolean column
    df['licensed'] = df['asset_id'].isin(licensed_asset_ids)

    # Add media_component_id column
    df['media_component_id'] = df['asset_id'].map(asset_to_media_component)

    # Add video availability column (True if available, False if blocked/unavailable)
    if 'video_id' in df.columns:
        msg = "Checking video availability"
        check_stopped(msg)
        if status_callback:
            status_callback(msg)
        available_map = check_videos_available_batch(df['video_id'].tolist(), youtube, check_stopped=check_stopped)
        df['video_available'] = df['video_id'].map(available_map)
    else:
        df['video_available'] = True  # Default to True if no video_id column

    # -----------------------------------------------------------------------
    # Engineer features and score with the AutoGluon challenger
    # -----------------------------------------------------------------------
    msg = "Engineering features (duration + fuzzy title matching)"
    check_stopped(msg)
    if status_callback:
        status_callback(msg)

    features = engineer_features(df)

    msg = "Scoring claims with AutoGluon"
    check_stopped(msg)
    if status_callback:
        status_callback(msg)

    predict_started = time.time()
    df['rating'] = predictor.predict_proba(features[AG_FEATURES]).iloc[:, 1].to_numpy()
    telemetry['predict_seconds'] = round(time.time() - predict_started, 2)
    telemetry['rows'] = len(df)

    # Trey's operational three-way decision, using the calibrated threshold
    df['action'] = np.where(
        df['rating'] >= AUTO_YES_THRESHOLD, 'Auto Yes',
        np.where(df['rating'] >= review_threshold, 'Human Review', 'Auto No'))

    # Exact duration matches are never auto-rejected without a human look
    df.loc[(df['action'] == 'Auto No') & (features['duration_diff_sec'] == 0),
           'action'] = 'Human Review'

    # Licensed assets and unavailable videos are always rejected
    rejected = (df['video_available'] == False) | (df['licensed'] == True)
    df.loc[rejected, 'rating'] = 0.0
    df.loc[rejected, 'action'] = 'Auto No'

    check_stopped("before saving predictions")
    if status_callback:
        status_callback("Saving predictions")

    # Save predictions
    df.to_csv(args.prediction_output, index=False)

    # Resource footprint, so production behaviour is observable rather than
    # inferred. peak_rss_mb is the high-water mark for the whole process.
    telemetry['total_seconds'] = round(time.time() - run_started, 2)
    telemetry['peak_rss_mb'] = peak_rss_mb()
    telemetry['actions'] = df['action'].value_counts().to_dict()
    return telemetry


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Score claims with Trey's pretrained AutoGluon model")
    parser.add_argument('--prediction-input', required=True, help='Input CSV for prediction')
    parser.add_argument('--prediction-output', default=f'ml_enriched_claims_{datetime.now().strftime("%Y%m%d%H%M")}.csv', help='Output CSV')

    args = parser.parse_args()

    main(args)

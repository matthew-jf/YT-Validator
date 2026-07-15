import argparse
from datetime import datetime
import json
import os
import sys
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
MODEL_DIR = REPO_ROOT / 'trey_pipeline' / 'models'

sys.path.insert(0, str(REPO_ROOT / 'trey_pipeline' / 'ml_pipeline'))
from feature_utils import engineer_features, AG_FEATURES

AUTO_YES_THRESHOLD = 0.97

# Columns engineer_features / the model need, beyond asset_id used for
# the licensed & media-component lookups.
REQUIRED_INPUT_COLUMNS = [
    'asset_id', 'video_title', 'asset_title', 'video_duration_sec',
    'duration_seconds', 'matching_duration', 'longest_match',
]


class TaskStoppedError(Exception):
    """Custom exception to indicate task was stopped."""
    pass


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

    # -----------------------------------------------------------------------
    # Load pretrained model artifacts
    # -----------------------------------------------------------------------
    msg = "Loading AutoGluon challenger model"
    check_stopped(msg)
    if status_callback:
        status_callback(msg)

    predictor = TabularPredictor.load(str(MODEL_DIR / 'ag_challenger'))
    with open(MODEL_DIR / 'ag_threshold.json') as f:
        review_threshold = json.load(f)['threshold']

    # -----------------------------------------------------------------------
    # Load licensed assets and asset->media_component mapping
    # -----------------------------------------------------------------------
    licensed_df = pandas.read_csv('Licensed.csv')
    licensed_asset_ids = set(licensed_df['asset_id'].dropna().unique())

    assets_media_df = pandas.read_csv('assets_single_media_component.csv')
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
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '')

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

    df['rating'] = predictor.predict_proba(features[AG_FEATURES]).iloc[:, 1].to_numpy()

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


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Score claims with Trey's pretrained AutoGluon model")
    parser.add_argument('--prediction-input', required=True, help='Input CSV for prediction')
    parser.add_argument('--prediction-output', default=f'ml_enriched_claims_{datetime.now().strftime("%Y%m%d%H%M")}.csv', help='Output CSV')

    args = parser.parse_args()

    main(args)

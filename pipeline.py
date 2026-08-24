import argparse
from datetime import datetime
import pandas
import numpy as np
from sklearn import base
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
import copy
import os
import warnings

from tqdm import tqdm
from googleapiclient.discovery import build
from helpers import load_env

load_env(["YT_API_KEY"])

# For checking YouTube video availability
youtube = build('youtube', 'v3', developerKey=os.environ["YT_API_KEY"])

# Suppress warnings
warnings.filterwarnings('ignore')

# Claims with these no_codes are excluded from training
excluded_codes = ['L', 'V', 'N', 'X']

claim_kind = ['VIDEO_MATCHAUDIOVISUAL', 'VIDEO_MATCHVISUAL', 'AUDIO_MATCHAUDIO', 'SHORTS_IN_PRODUCTAUDIO', 'WEB_UPLOAD_BY_OWNERAUDIOVISUAL', 'DESCRIPTIVE_SEARCHAUDIOVISUAL', 'CMS_UPLOADAUDIOVISUAL']
content_type = ['UGC', 'SONG_UGC', 'PARTNER_UPLOADED']


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
    # Training data
    # -----------------------------------------------------------------------
    if not os.path.exists('YT.csv'):
        if not args.training_data:
            raise ValueError("Training data is required to re-create YT.csv")

        msg = "Loading training data"
        check_stopped(msg)
        if status_callback:
            status_callback(msg)

        df = pandas.read_csv(args.training_data,
            dtype=dict(views='Int32', matching_duration='Int32',
                       longest_match='Int32', video_duration_sec='Int32'))
        df = df[df.verdict != 'U']
        df = df[~df.no_code.isin(excluded_codes)]
        df.verdict = np.array(df.verdict == 'Y', dtype=int)

        # Balanced sample: 100K per verdict class (200K total) to ensure class
        # balance. If a class has fewer than PER_CLASS_N rows, take all of them.
        PER_CLASS_N = 100000
        df = pandas.concat(
            [g.sample(n=min(PER_CLASS_N, len(g)), random_state=0)
             for _, g in df.groupby('verdict')]
        ).reset_index(drop=True)

        # Create claim feature and select columns
        df['claim'] = df.claim_origin + df.claim_type
        df = df[[
            'views',
            'matching_duration',
            'longest_match',
            'video_duration_sec',
            'verdict',
            'claim',
            'content_type'
        ]]

        # One-hot encode claim types
        for s in claim_kind:
            df[s] = np.array(df.claim == s, dtype=int)
        df = df.drop(columns='claim')

        # One-hot encode content types
        for ct in content_type:
            df[ct] = np.array(df.content_type == ct, dtype=int)
        df = df.drop(columns='content_type')

        df = df.fillna(0)
        df.to_csv('YT.csv', index=False)

    check_stopped("before model training")
    if status_callback:
        status_callback(f"Training XGBoost model & {'NOT' if args.skip_validation else ''} cross-validating")

    # -----------------------------------------------------------------------
    # Train model
    # -----------------------------------------------------------------------
    df = pandas.read_csv('YT.csv')
    df, y = df.drop(columns='verdict'), df.verdict

    # Handle class imbalance the XGBoost-native way. With the 100K/100K balanced
    # sample this is ~1.0; kept explicit so it stays correct if a class is smaller
    # than PER_CLASS_N.
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    soln = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=0,
    )

    if not args.skip_validation:

        if status_callback:
            status_callback("Performing cross-validation")

        for _ in range(4):
            test = np.random.permutation(len(df))
            test = test[:len(df) // 4]
            test = np.array([i in test for i in range(len(df))])

            soln.fit(df[~test], y[~test])
            valid = soln.predict_proba(df[test])
            valid = valid[:, 1]
            print(balanced_accuracy_score(y[test], valid > 1/2))
            soln = base.clone(soln)

    soln.fit(df, y)

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
    df = pandas.read_csv(args.prediction_input,
        dtype=dict(views='Int32', matching_duration='Int32',
                   longest_match='Int32', video_duration_sec='Int32'))
    df2 = copy.copy(df)

    # Add licensed boolean column
    df['licensed'] = df['asset_id'].isin(licensed_asset_ids)

    # Add media_component_id column
    df['media_component_id'] = df['asset_id'].map(asset_to_media_component)

    # Add video availability column (True if available, False if blocked/unavailable)
    if 'video_id' in df.columns:
        msg = "Checking video availability"
        check_stopped(msg)
        available_map = check_videos_available_batch(df['video_id'].tolist(), youtube, check_stopped=check_stopped)
        df['video_available'] = df['video_id'].map(available_map)
    else:
        df['video_available'] = True  # Default to True if no video_id column

    # Prepare features
    df2['claim'] = df2.claim_origin + df2.claim_type
    df2 = df2[[
        'views',
        'matching_duration',
        'longest_match',
        'video_duration_sec',
        'claim',
        'content_type'
    ]]

    # One-hot encode claim types (using same categories from training)
    for s in claim_kind:
        df2[s] = np.array(df2.claim == s, dtype=int)
    df2 = df2.drop(columns='claim')

    # One-hot encode content types
    for ct in content_type:
        df2[ct] = np.array(df2.content_type == ct, dtype=int)
    df2 = df2.drop(columns='content_type')

    df2 = df2.fillna(0)

    check_stopped("before predictions")
    if status_callback:
        status_callback("Making predictions")

    # Make predictions
    valid = soln.predict_proba(df2)
    valid = valid[:, 1]
    df['rating'] = valid

    # Set rating to 0 for unavailable videos
    if 'video_available' in df.columns:
        df.loc[df['video_available'] == False, 'rating'] = 0

    # Set rating to 0 for licensed assets
    df.loc[df['licensed'] == True, 'rating'] = 0

    # Save predictions
    df.to_csv(args.prediction_output, index=False)


class TaskStoppedError(Exception):
    """Custom exception to indicate task was stopped."""
    pass


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process claims data and train classifier')
    parser.add_argument('--training-data', default='./data/export_all_claims_202505211438.csv', help='Training data CSV')
    parser.add_argument('--prediction-input', required=True, help='Input CSV for prediction')
    parser.add_argument('--prediction-output', default=f'ml_enriched_claims_{datetime.now().strftime("%Y%m%d%H%M")}.csv', help='Output CSV')
    parser.add_argument('--skip-validation', action='store_true', help='Skip e.g. cross-validation, etc.')

    args = parser.parse_args()

    main(args)

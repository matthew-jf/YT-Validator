import argparse
from datetime import datetime
import pandas
import numpy as np
from sklearn import (
    neighbors,
    base
)
import copy
from transformers import pipeline
from tqdm import tqdm
import warnings
import os
import time

# For checking YouTube video availability
from pytube import YouTube

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", 
                     model="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")

# Load category descriptions
categories_df = pandas.read_csv('no_codes - Sheet1.csv')

# Filter out unwanted categories
excluded_codes = ['L', 'V', 'N', 'X']
categories_df = categories_df[~categories_df['code'].isin(excluded_codes)]
candidate_labels = categories_df['long_desc'].tolist()
codes = categories_df['code'].tolist()

# Define zeroshot_cols and claim_kind outside conditional block
zeroshot_cols = [f'zeroshot_score_{code}' for code in codes]
claim_kind = ['VIDEO_MATCHAUDIOVISUAL', 'VIDEO_MATCHVISUAL', 'AUDIO_MATCHAUDIO', 'SHORTS_IN_PRODUCTAUDIO', 'WEB_UPLOAD_BY_OWNERAUDIOVISUAL', 'DESCRIPTIVE_SEARCHAUDIOVISUAL', 'CMS_UPLOADAUDIOVISUAL']
content_type = ['UGC', 'SONG_UGC', 'PARTNER_UPLOADED']

# Function to check if a YouTube video is available
def check_video_available(video_id, io_rate_limit=1):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(url)
        _ = yt.title  # Accessing title to trigger fetch
        time.sleep(io_rate_limit)  # Add 1 second delay between requests
        return True
    except Exception as e:
        time.sleep(io_rate_limit)  # Add 1 second delay even on errors
        return False

def add_zeroshot_features(df, batch_size=100):
    df['channel_display_name'] = df['channel_display_name'].fillna('')
    df['video_title'] = df['video_title'].fillna('')
    texts = (df['channel_display_name'] + ' ' + df['video_title']).tolist()
    
    # Initialize columns
    for code in codes:
        df[f'zeroshot_score_{code}'] = 0.0
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot classification"):
        batch_texts = texts[i:i+batch_size]
        results = classifier(batch_texts, candidate_labels, multi_label=False)
        
        for j, result in enumerate(results):
            score_dict = dict(zip(result['labels'], result['scores']))
            for code, desc in zip(codes, candidate_labels):
                df.loc[i+j, f'zeroshot_score_{code}'] = score_dict[desc]
    
    return df


def main(args, status_callback=None):

    # Load training data and create YT.csv if it doesn't exist
    if not os.path.exists('YT.csv'):
        
        if status_callback:
            status_callback("Loading training data")
    
        df = pandas.read_csv(args.training_data,
            dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
        df = df[df.verdict != 'U']
        df.verdict = np.array(df.verdict == 'Y', dtype=int)

        # Add zero-shot features to training data
        df = add_zeroshot_features(df)

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
        ] + zeroshot_cols]

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

    if status_callback:
        status_callback("Training kNN model & cross-validating")

    # Train model
    df = pandas.read_csv('YT.csv')
    df, y = df.drop(columns='verdict'), df.verdict
    soln = neighbors.KNeighborsClassifier(n_neighbors=11, p=1)

    if not args.skip_validation:

        if status_callback:
            status_callback("Performing cross-validation")
        
        for _ in range(4):
            test = np.random.permutation(len(df))
            test = test[:len(df) // 4]
            test = np.array([i in test for i in range(len(df))])

            soln.fit(df[~test], y[~test])
            valid = soln.predict_proba(df[test])
            valid = valid[:,1]
            print(sum((valid > 1/2) == y[test]) / sum(test))
            soln = base.clone(soln)

    soln.fit(df, y)

    # Load licensed assets
    licensed_df = pandas.read_csv('Licensed.csv')
    licensed_asset_ids = set(licensed_df['asset_id'].dropna().unique())

    # Process unprocessed claims
    df = pandas.read_csv(args.prediction_input,
        dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
    df2 = copy.copy(df)

    # Add licensed boolean column
    df['licensed'] = df['asset_id'].isin(licensed_asset_ids)

    # Add video availability column (True if available, False if blocked/unavailable)
    if 'video_id' in df.columns:
        if status_callback:
            desc = "Checking video availability"
            status_callback(desc)
        tqdm.pandas(desc=desc)
        df['video_available'] = df['video_id'].progress_apply(
            lambda vid: check_video_available(vid, io_rate_limit=args.io_rate_limit))
    else:
        df['video_available'] = True  # Default to True if no video_id column

    # Add zero-shot features to unprocessed data
    df2 = add_zeroshot_features(df2)

    # Prepare features
    df2['claim'] = df2.claim_origin + df2.claim_type
    df2 = df2[[
        'views',
        'matching_duration',
        'longest_match',
        'video_duration_sec',
        'claim',
        'content_type'
    ] + zeroshot_cols]

    # One-hot encode claim types (using same categories from training)
    for s in claim_kind:
        df2[s] = np.array(df2.claim == s, dtype=int)
    df2 = df2.drop(columns='claim')

    # One-hot encode content types
    for ct in content_type:
        df2[ct] = np.array(df2.content_type == ct, dtype=int)
    df2 = df2.drop(columns='content_type')

    df2 = df2.fillna(0)

    if status_callback:
        status_callback("Making predictions")

    # Make predictions
    valid = soln.predict_proba(df2)
    valid = valid[:,1]
    df['rating'] = valid

    # Set rating to 0 for unavailable videos
    if 'video_available' in df.columns:
        df.loc[df['video_available'] == False, 'rating'] = 0

    # Set rating to 0 for licensed assets
    df.loc[df['licensed'] == True, 'rating'] = 0

    # Add zeroshot features to the output dataframe
    for col in zeroshot_cols:
        df[col] = df2[col]

    # Save predictions
    df.to_csv(args.prediction_output, index=False)


if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process claims data and train classifier')
    parser.add_argument('--training-data', default='./data/export_all_claims_202505211438.csv', help='Training data CSV')
    parser.add_argument('--prediction-input', required=True, help='Input CSV for prediction')
    parser.add_argument('--prediction-output', default=f'export_all_claims_{datetime.now().strftime("%Y%m%d%H%M")}.csv', help='Output CSV')
    parser.add_argument('--io-rate-limit',  type=float, default=1, help='Rate limit for I/O operations (seconds)')
    parser.add_argument('--skip-validation', action='store_true', help='Skip feg. cross-validation, etc.')

    args = parser.parse_args()

    main(args)
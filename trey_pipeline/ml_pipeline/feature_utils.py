# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "thefuzz"]
# ///

import pandas as pd
from thefuzz import fuzz

def engineer_features(df):
    """Applies duration mathematics and fuzzy text matching to the raw dataset."""
    df = df.copy()
    
    # 1. Duration Features
    df['video_duration_sec'] = pd.to_numeric(df['video_duration_sec'], errors='coerce')
    df['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce')
    df['duration_diff_sec'] = abs(df['video_duration_sec'] - df['duration_seconds'])
    df['duration_ratio'] = df['video_duration_sec'] / (df['duration_seconds'] + 0.001)

    # 2. Text Similarity Features
    df['video_title'] = df['video_title'].fillna('')
    df['asset_title'] = df['asset_title'].fillna('')

    df['title_fuzzy_ratio'] = df.apply(
        lambda row: fuzz.ratio(str(row['video_title']), str(row['asset_title'])), axis=1
    )
    df['title_token_sort_ratio'] = df.apply(
        lambda row: fuzz.token_sort_ratio(str(row['video_title']), str(row['asset_title'])), axis=1
    )
    df['title_token_set_ratio'] = df.apply(
        lambda row: fuzz.token_set_ratio(str(row['video_title']), str(row['asset_title'])), axis=1
    )

    # 3. Clean numericals
    df['matching_duration'] = pd.to_numeric(df['matching_duration'], errors='coerce')
    df['longest_match'] = pd.to_numeric(df['longest_match'], errors='coerce')
    
    return df

BASE_FEATURES = [
    'duration_diff_sec', 'duration_ratio', 'title_fuzzy_ratio', 
    'title_token_sort_ratio', 'title_token_set_ratio', 
    'matching_duration', 'longest_match'
]

AG_FEATURES = BASE_FEATURES + ['video_title', 'asset_title']

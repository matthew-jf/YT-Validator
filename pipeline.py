import pandas
import numpy as np
from sklearn import base
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
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
def check_video_available(video_id):
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(url)
        _ = yt.title  # Accessing title to trigger fetch
        time.sleep(0.5)  # Add 0.5 second delay between requests
        return True
    except Exception as e:
        time.sleep(0.5)  # Add 0.5 second delay even on errors
        return False

def add_zeroshot_features(df, batch_size=100):
    df = df.reset_index(drop=True)
    df['channel_display_name'] = df['channel_display_name'].fillna('')
    df['video_title'] = df['video_title'].fillna('')
    texts = (df['channel_display_name'] + ' ' + df['video_title']).tolist()

    scores = {code: np.zeros(len(texts), dtype=float) for code in codes}

    for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot classification"):
        batch_texts = texts[i:i+batch_size]
        results = classifier(batch_texts, candidate_labels, multi_label=False)
        for j, result in enumerate(results):
            score_dict = dict(zip(result['labels'], result['scores']))
            for code, desc in zip(codes, candidate_labels):
                scores[code][i+j] = score_dict[desc]

    for code in codes:
        df[f'zeroshot_score_{code}'] = scores[code]
    return df

# Load training data and create YT.csv if it doesn't exist
if not os.path.exists('YT.csv'):
    df = pandas.read_csv(r"/Users/matthew.jurewicz/Downloads/export_all_claims_202507241336.csv",
        dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
    df = df[df.verdict != 'U']
    df = df[~df.no_code.isin(excluded_codes)]
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

# Train model
df = pandas.read_csv('YT.csv')
df, y = df.drop(columns='verdict'), df.verdict
soln = make_pipeline(
    StandardScaler(),
    SGDClassifier(loss='log_loss', class_weight='balanced', random_state=0)
)
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

# Load licensed assets
licensed_df = pandas.read_csv('Licensed.csv')
licensed_asset_ids = set(licensed_df['asset_id'].dropna().unique())

# Load asset to media_component_id mapping
assets_media_df = pandas.read_csv('assets_single_media_component.csv')
asset_to_media_component = dict(zip(assets_media_df['asset_id'], assets_media_df['media_component_id']))

# Process unprocessed claims
df = pandas.read_csv(r"/Users/matthew.jurewicz/Downloads/export_unprocessed_claims_202509051550.csv",
    dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
df2 = copy.copy(df)

# Add licensed boolean column
df['licensed'] = df['asset_id'].isin(licensed_asset_ids)

# Add media_component_id column
df['media_component_id'] = df['asset_id'].map(asset_to_media_component)

# Add video availability column (True if available, False if blocked/unavailable)
if 'video_id' in df.columns:
    tqdm.pandas(desc="Checking video availability")
    df['video_available'] = df['video_id'].progress_apply(check_video_available)
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

# Make predictions
valid = soln.predict_proba(df2)
valid = valid[:, 1]
df['rating'] = valid

# Set rating to 0 for unavailable videos
if 'video_available' in df.columns:
    df.loc[df['video_available'] == False, 'rating'] = 0

# Set rating to 0 for licensed assets
df.loc[df['licensed'] == True, 'rating'] = 0

# Add zeroshot features to the output dataframe
for col in zeroshot_cols:
    df[col] = df2[col]

df.to_csv('export_unprocessed_claims_202509051550.csv', index=False)
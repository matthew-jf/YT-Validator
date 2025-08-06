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

# Load training data and create YT.csv if it doesn't exist
if not os.path.exists('YT.csv'):
    df = pandas.read_csv(r"/Users/matthew.jurewicz/Downloads/export_all_claims_202507241336.csv",
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
        'claim'
    ] + zeroshot_cols]

    # One-hot encode claim types
    for s in claim_kind:
        df[s] = np.array(df.claim == s, dtype=int)
    df = df.drop(columns='claim')
    df = df.fillna(0)
    df.to_csv('YT.csv', index=False)

# Train model
df = pandas.read_csv('YT.csv')
df, y = df.drop(columns='verdict'), df.verdict
soln = neighbors.KNeighborsClassifier(n_neighbors=11, p=1)
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

# Process unprocessed claims
df = pandas.read_csv(r"/Users/matthew.jurewicz/Downloads/export_unprocessed_claims_202507241337.csv",
    dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
df2 = copy.copy(df)

# Add zero-shot features to unprocessed data
df2 = add_zeroshot_features(df2)

# Prepare features
df2['claim'] = df2.claim_origin + df2.claim_type
df2 = df2[[
    'views',
    'matching_duration',
    'longest_match',
    'video_duration_sec',
    'claim'
] + zeroshot_cols]

# One-hot encode claim types (using same categories from training)
for s in claim_kind:
    df2[s] = np.array(df2.claim == s, dtype=int)
df2 = df2.drop(columns='claim')
df2 = df2.fillna(0)

# Make predictions
valid = soln.predict_proba(df2)
valid = valid[:,1]
df['rating'] = valid

# Add zeroshot features to the output dataframe
for col in zeroshot_cols:
    df[col] = df2[col]

df.to_csv('export_unprocessed_claims_202507241337.csv', index=False)
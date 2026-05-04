import pandas
import numpy as np
from sklearn import base
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
import copy
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

# For checking YouTube video availability
from pytube import YouTube

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# OpenAI / ChatGPT zero-shot setup
# ---------------------------------------------------------------------------
# Requires:  pip install openai
# Requires:  export OPENAI_API_KEY=...
OPENAI_MODEL = os.environ.get("OPENAI_ZEROSHOT_MODEL", "gpt-4o-mini")
OPENAI_MAX_WORKERS = int(os.environ.get("OPENAI_ZEROSHOT_WORKERS", "8"))
OPENAI_BATCH_SIZE = int(os.environ.get("OPENAI_ZEROSHOT_BATCH_SIZE", "20"))
OPENAI_MAX_RETRIES = 3

client = OpenAI()  # picks up OPENAI_API_KEY from env

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
    except Exception:
        time.sleep(0.5)  # Add 0.5 second delay even on errors
        return False


# ---------------------------------------------------------------------------
# Zero-shot classification via ChatGPT (batched)
# ---------------------------------------------------------------------------
def _build_zeroshot_batch_prompt(texts, codes, candidate_labels):
    labels_block = "\n".join(
        f"- {code}: {desc}" for code, desc in zip(codes, candidate_labels)
    )
    # JSON-encode each text so newlines/quotes inside titles don't break parsing.
    items_block = "\n".join(
        f"{i + 1}. {json.dumps(text, ensure_ascii=False)}"
        for i, text in enumerate(texts)
    )
    return (
        "You are a zero-shot text classifier. For EACH numbered text below, "
        "assign probabilities to the category codes. Probabilities for a "
        "single text MUST be non-negative and sum to 1.0 across all codes. "
        "Pick the single best-fitting code as the highest-probability one; "
        "do not split mass evenly unless the text is genuinely ambiguous.\n\n"
        f"Categories:\n{labels_block}\n\n"
        f"Texts:\n{items_block}\n\n"
        'Return ONLY a JSON object of the form '
        '{"results": [{"id": <int>, "scores": {"<code>": <prob>, ...}}, ...]} '
        f"with exactly one entry per input id (1..{len(texts)}), "
        "and every category code present in every entry."
    )


def _classify_batch(texts, codes, candidate_labels):
    """Classify a batch of texts in a single API call.

    Returns list[dict[code, prob]] aligned with input order.
    """
    uniform = {code: 1.0 / len(codes) for code in codes}
    if not texts:
        return []

    last_err = None
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role": "user",
                    "content": _build_zeroshot_batch_prompt(texts, codes, candidate_labels),
                }],
                response_format={"type": "json_object"},
                temperature=0,
            )
            payload = json.loads(resp.choices[0].message.content)
            items = payload.get("results", [])

            # Index returned items by their reported id (1-based).
            by_id = {}
            for it in items:
                if not isinstance(it, dict):
                    continue
                rid = it.get("id")
                raw = it.get("scores", {})
                if rid is None or not isinstance(raw, dict):
                    continue
                scored = {code: float(raw.get(code, 0.0)) for code in codes}
                total = sum(scored.values())
                if total > 0:
                    scored = {code: v / total for code, v in scored.items()}
                else:
                    scored = dict(uniform)
                try:
                    by_id[int(rid)] = scored
                except (TypeError, ValueError):
                    continue

            # Align with input order; fall back to uniform for empty/missing.
            results = []
            for i, text in enumerate(texts):
                if not text or not str(text).strip():
                    results.append(dict(uniform))
                else:
                    results.append(by_id.get(i + 1, dict(uniform)))
            return results
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)

    print(f"[zeroshot] batch of {len(texts)} failed after "
          f"{OPENAI_MAX_RETRIES} retries: {last_err}")
    return [dict(uniform) for _ in texts]


def add_zeroshot_features(df, batch_size=OPENAI_BATCH_SIZE,
                          max_workers=OPENAI_MAX_WORKERS):
    df = df.reset_index(drop=True)
    df['channel_display_name'] = df['channel_display_name'].fillna('')
    df['video_title'] = df['video_title'].fillna('')
    texts = (df['channel_display_name'] + ' ' + df['video_title']).tolist()

    scores = {code: np.zeros(len(texts), dtype=float) for code in codes}

    # Slice into batches; remember each batch's starting offset.
    batches = [
        (start, texts[start:start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_start = {
            executor.submit(_classify_batch, batch_texts, codes, candidate_labels): start
            for start, batch_texts in batches
        }
        with tqdm(total=len(texts),
                  desc="Zero-shot classification (ChatGPT, batched)") as pbar:
            for future in as_completed(future_to_start):
                start = future_to_start[future]
                batch_results = future.result()
                for j, result in enumerate(batch_results):
                    for code in codes:
                        scores[code][start + j] = result[code]
                pbar.update(len(batch_results))

    for code in codes:
        df[f'zeroshot_score_{code}'] = scores[code]
    return df


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
if not os.path.exists('YT.csv'):
    df = pandas.read_csv(
        r"/Users/matthew.jurewicz/Downloads/export_all_claims_202507241336.csv",
        dtype=dict(views='Int32', matching_duration='Int32',
                   longest_match='Int32', video_duration_sec='Int32'))
    df = df[df.verdict != 'U']
    df = df[~df.no_code.isin(excluded_codes)]
    df.verdict = np.array(df.verdict == 'Y', dtype=int)

    # Balanced sample: 100K per verdict class (200K total) to limit zero-shot
    # API calls / cost while ensuring class balance. If a class has fewer than
    # PER_CLASS_N rows, take all of them.
    PER_CLASS_N = 100000
    df = pandas.concat(
        [g.sample(n=min(PER_CLASS_N, len(g)), random_state=0)
         for _, g in df.groupby('verdict')]
    ).reset_index(drop=True)

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

# ---------------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Load licensed assets and asset->media_component mapping
# ---------------------------------------------------------------------------
licensed_df = pandas.read_csv('Licensed.csv')
licensed_asset_ids = set(licensed_df['asset_id'].dropna().unique())

assets_media_df = pandas.read_csv('assets_single_media_component.csv')
asset_to_media_component = dict(zip(assets_media_df['asset_id'],
                                    assets_media_df['media_component_id']))

# ---------------------------------------------------------------------------
# Process unprocessed claims
# ---------------------------------------------------------------------------
df = pandas.read_csv(
    r"/Users/matthew.jurewicz/Downloads/export_unprocessed_claims_202509051550.csv",
    dtype=dict(views='Int32', matching_duration='Int32',
               longest_match='Int32', video_duration_sec='Int32'))
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
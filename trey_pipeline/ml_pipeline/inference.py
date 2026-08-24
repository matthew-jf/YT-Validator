# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "numpy", "xgboost", "thefuzz", "autogluon"]
# ///

import os
import json
from pathlib import Path
import pandas as pd
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from feature_utils import engineer_features, BASE_FEATURES, AG_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def run_inference(
    data_path=PROJECT_ROOT / "data" / "unprocessed_claims.csv",
    model_dir=PROJECT_ROOT / "models",
    output_path=PROJECT_ROOT / "data" / "output_claims.csv",
):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, engine='python', on_bad_lines='skip', encoding='utf-8-sig')
    df.columns = df.columns.astype(str).str.strip()
    
    verdict_col = 'Ver-dict' if 'Ver-dict' in df.columns else [c for c in df.columns if 'verdict' in c.lower()][0]
    df.rename(columns={verdict_col: 'Actual_Verdict'}, inplace=True)
    df['Actual_Verdict'] = df['Actual_Verdict'].astype(str).str.strip()
    
    print("Engineering features...")
    X_raw = engineer_features(df.iloc[:, :69])
    
    print("Loading Models...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(os.path.join(model_dir, "xgb_baseline.json"))
    ag_predictor = TabularPredictor.load(os.path.join(model_dir, "ag_challenger"))
    
    with open(os.path.join(model_dir, "ag_threshold.json"), "r") as f:
        ag_threshold = json.load(f)["threshold"]

    # Generate XGBoost Outputs
    df['XGB_Confidence'] = xgb_model.predict_proba(X_raw[BASE_FEATURES].astype(float))[:, 1]
    df['XGB_Action'] = df['XGB_Confidence'].apply(
        lambda x: 'Auto Yes' if x >= 0.97 else ('Human Review' if x >= 0.60 else 'Auto No')
    )
    df.loc[(df['XGB_Action'] == 'Auto No') & (X_raw['duration_diff_sec'] == 0), 'XGB_Action'] = 'Human Review'

    # Generate AutoGluon Outputs
    df['AG_Confidence'] = ag_predictor.predict_proba(X_raw[AG_FEATURES]).iloc[:, 1]
    df['AG_Action'] = df['AG_Confidence'].apply(
        lambda x: 'Auto Yes' if x >= 0.97 else ('Human Review' if x >= ag_threshold else 'Auto No')
    )
    df.loc[(df['AG_Action'] == 'Auto No') & (X_raw['duration_diff_sec'] == 0), 'AG_Action'] = 'Human Review'

    # Reorder columns to pull actions to the front
    cols = df.columns.tolist()
    front_cols = ['Actual_Verdict', 'XGB_Action', 'XGB_Confidence', 'AG_Action', 'AG_Confidence']
    for c in front_cols: 
        if c in cols: cols.remove(c)
    
    df[front_cols + cols].to_csv(output_path, index=False)
    print(f"Inference complete! Saved to {output_path}")

if __name__ == "__main__":
    run_inference()

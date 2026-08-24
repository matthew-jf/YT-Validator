# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "numpy", "xgboost", "thefuzz", "autogluon", "scikit-learn"]
# ///

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from autogluon.tabular import TabularPredictor
from feature_utils import engineer_features, BASE_FEATURES, AG_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTOGLUON_TIME_LIMIT_SECONDS = .5 * 60 * 60

def run_training(data_path=PROJECT_ROOT / "data" / "all_claims.csv", model_dir=PROJECT_ROOT / "models"):
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load and slice data
    df = pd.read_csv(data_path, engine='python', on_bad_lines='skip', encoding='utf-8-sig')
    df.columns = df.columns.astype(str).str.strip()
    
    verdict_col = 'Ver-dict' if 'Ver-dict' in df.columns else [c for c in df.columns if 'verdict' in c.lower()][0]
    valid_cases = df[df['verdict'].isin(['Y', 'N'])].copy()
    
    y = valid_cases[verdict_col].map({'Y': 1, 'N': 0})
    X_raw = valid_cases.drop(columns=[verdict_col])
    
    print("Engineering features...")
    X_processed = engineer_features(X_raw)
    
    # 2. Train/Test Split, grouped by feature-vector
    # different claims (even from different assets) often share all 9 features. standard train_test_split scattered
    # those duplicates across the train/test boundary, so 26.3% of the test set was already in
    # training and the score measured memory. Grouping keeps twins one side.
    groups = X_processed[AG_FEATURES].astype(str).agg("|".join, axis=1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_processed, y, groups=groups))
    X_train, X_test = X_processed.iloc[train_idx], X_processed.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 3. Train XGBoost
    print("Training XGBoost Baseline...")
    weight = (y_train == 0).sum() / (y_train == 1).sum() * 3.0 # Asymmetric harshness
    xgb_path = os.path.join(model_dir, "xgb_baseline.json")
    xgb_model = xgb.XGBClassifier()
    if os.path.exists(xgb_path):
        print("Loading existing XGBoost Baseline...")
        xgb_model.load_model(xgb_path)
    else:
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss', max_depth=5,
            learning_rate=0.1, n_estimators=200, random_state=42, scale_pos_weight=weight
        )
        xgb_model.fit(X_train[BASE_FEATURES].astype(float), y_train)
        xgb_model.save_model(xgb_path)
    
    # Calculate XGB False Negatives to calibrate AutoGluon
    y_pred_xgb = xgb_model.predict_proba(X_test[BASE_FEATURES].astype(float))[:, 1]
    xgb_actions = np.where(y_pred_xgb >= 0.97, 'Auto Yes', np.where(y_pred_xgb >= 0.60, 'Human Review', 'Auto No'))
    xgb_actions[(xgb_actions == 'Auto No') & (X_test['duration_diff_sec'] == 0)] = 'Human Review'
    target_fns = ((y_test == 1) & (xgb_actions == 'Auto No')).sum()

    # 4. Train AutoGluon
    print("Training AutoGluon Challenger Stack...")
    ag_train = X_train[AG_FEATURES].copy()
    ag_train['target'] = y_train
    ag_train['weight'] = np.where(y_train == 1, weight, 1.0)
    
    ag_path = os.path.join(model_dir, "ag_challenger")
    if os.path.exists(os.path.join(ag_path, "predictor.pkl")):
        print("Loading existing AutoGluon Challenger Stack...")
        predictor = TabularPredictor.load(ag_path)
    else:
        predictor = TabularPredictor(label='target', eval_metric='log_loss', sample_weight='weight', path=ag_path)
        predictor.fit(
            train_data=ag_train,
            presets='high_quality',
            time_limit=AUTOGLUON_TIME_LIMIT_SECONDS,
            dynamic_stacking=False,
        )
    
    # 5. Calibrate AutoGluon Threshold
    print("Calibrating AutoGluon inference thresholds...")
    y_pred_ag = predictor.predict_proba(X_test[AG_FEATURES]).iloc[:, 1]
    best_threshold, min_fns = 0.60, float('inf')
    
    for candidate in np.linspace(0.10, 0.70, 61):
        ag_actions = np.where(y_pred_ag >= 0.97, 'Auto Yes', np.where(y_pred_ag >= candidate, 'Human Review', 'Auto No'))
        ag_actions[(ag_actions == 'Auto No') & (X_test['duration_diff_sec'] == 0)] = 'Human Review'
        current_fns = ((y_test == 1) & (ag_actions == 'Auto No')).sum()
        
        if current_fns <= target_fns and current_fns < min_fns:
            min_fns = current_fns
            best_threshold = candidate
            
    with open(os.path.join(model_dir, "ag_threshold.json"), "w") as f:
        json.dump({"threshold": best_threshold}, f)
        
    print(f"Training Complete. Models saved to {model_dir}/")

if __name__ == "__main__":
    run_training()

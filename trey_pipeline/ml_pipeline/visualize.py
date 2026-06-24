# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "matplotlib", "seaborn", "scikit-learn"]
# ///

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def evaluate_and_visualize(
    data_path=PROJECT_ROOT / "data" / "clean_all_claims.csv",
    output_path=PROJECT_ROOT / "data" / "workload_reduction.png",
):
    df = pd.read_csv(data_path)
    valid_cases = df[df['Actual_Verdict'].isin(['Y', 'N'])].copy()
    
    if valid_cases.empty:
        print("No historical ground truth ('Y' or 'N') found in the file. Skipping visualization.")
        return
        
    valid_cases['Actual_Binary'] = valid_cases['Actual_Verdict'].map({'Y': 1, 'N': 0})
    
    def calc_metrics(action_col):
        auto = valid_cases[valid_cases[action_col] != 'Human Review'].copy()
        if auto.empty: return 0, 0, 0
        auto['Pred'] = auto[action_col].map({'Auto Yes': 1, 'Auto No': 0})
        
        rate = len(auto) / len(valid_cases) * 100
        acc = accuracy_score(auto['Actual_Binary'], auto['Pred']) * 100
        fns = ((valid_cases['Actual_Binary'] == 1) & (valid_cases[action_col] == 'Auto No')).sum()
        return rate, acc, fns

    xgb_rate, xgb_acc, xgb_fns = calc_metrics('XGB_Action')
    ag_rate, ag_acc, ag_fns = calc_metrics('AG_Action')
    
    print("\n" + "="*60)
    print(" HEAD-TO-HEAD PERFORMANCE REPORT")
    print("="*60)
    print(f"XGBoost   | Automation: {xgb_rate:.2f}% | Accuracy: {xgb_acc:.2f}% | False Negatives: {xgb_fns}")
    print(f"AutoGluon | Automation: {ag_rate:.2f}% | Accuracy: {ag_acc:.2f}% | False Negatives: {ag_fns}")
    print("="*60)

    # Workload Reduction Chart
    plt.figure(figsize=(10, 6))
    actions_df = pd.melt(valid_cases[['XGB_Action', 'AG_Action']], var_name='Model', value_name='Action')
    sns.countplot(data=actions_df, x='Model', hue='Action', palette='muted')
    plt.title('Manual Workload Reduction: XGBoost vs AutoGluon')
    plt.ylabel('Number of Claims')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to '{output_path}'")

if __name__ == "__main__":
    evaluate_and_visualize()

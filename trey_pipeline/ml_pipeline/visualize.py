# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "matplotlib", "seaborn", "scikit-learn", "numpy"]
# ///

#visualize.py
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================================
# 1. MODEL EVALUATION (XGBOOST VS AUTOGLUON)
# ==========================================
def evaluate_models(
    preds_path=PROJECT_ROOT / "data" / "output_claims.csv",
    truth_path=PROJECT_ROOT / "data" / "unprocessed_claims_Matt_MCN_MARCH1Rev.csv",
    output_path=PROJECT_ROOT / "data" / "workload_reduction.png",
):
    print("\n" + "="*65)
    print("           MODEL PERFORMANCE & WORKLOAD REDUCTION")
    print("="*65)
    
    try:
        preds_df = pd.read_csv(preds_path)
        truth_df = pd.read_csv(truth_path)
    except FileNotFoundError as e:
        print(f"File not found: {e}. Skipping model evaluation.")
        return

    # Merge predictions with ground truth on 'video_id'
    if 'video_id' not in preds_df.columns or 'video_id' not in truth_df.columns:
        print("Error: 'video_id' column must be present in both files to merge them.")
        return
        
    df = pd.merge(preds_df, truth_df, on='video_id', how='inner')
    
    # Filter to valid ground truth
    valid_cases = df[df['verdict'].isin(['Y', 'N'])].copy()
    
    if valid_cases.empty:
        print("No valid historical ground truth ('Y' or 'N') found after merging.")
        return
        
    valid_cases['Actual_Binary'] = valid_cases['verdict'].map({'Y': 1, 'N': 0})
    
    def calc_metrics(action_col):
        if action_col not in valid_cases.columns:
            return 0, 0, 0
            
        auto = valid_cases[valid_cases[action_col] != 'Human Review'].copy()
        if auto.empty: return 0, 0, 0
        
        auto['Pred'] = auto[action_col].map({'Auto Yes': 1, 'Auto No': 0})
        rate = len(auto) / len(valid_cases) * 100
        acc = accuracy_score(auto['Actual_Binary'], auto['Pred']) * 100
        fns = ((valid_cases['Actual_Binary'] == 1) & (valid_cases[action_col] == 'Auto No')).sum()
        return rate, acc, fns

    # Calculate for both models
    xgb_rate, xgb_acc, xgb_fns = calc_metrics('XGB_Action')
    ag_rate, ag_acc, ag_fns = calc_metrics('AG_Action')
    
    # Generate Text Report
    print(f"XGBoost   | Automation: {xgb_rate:.2f}% | Accuracy: {xgb_acc:.2f}% | False Negatives: {xgb_fns}")
    print(f"AutoGluon | Automation: {ag_rate:.2f}% | Accuracy: {ag_acc:.2f}% | False Negatives: {ag_fns}")
    print("="*65)

    # Generate Workload Reduction Chart
    if 'XGB_Action' in valid_cases.columns and 'AG_Action' in valid_cases.columns:
        plt.figure(figsize=(10, 6))
        actions_df = pd.melt(valid_cases[['XGB_Action', 'AG_Action']], var_name='Model', value_name='Action')
        
        sns.countplot(
            data=actions_df, 
            x='Model', 
            hue='Action', 
            palette='muted', 
            order=['XGB_Action', 'AG_Action'],
            hue_order=['Auto Yes', 'Human Review', 'Auto No']
        )
        
        plt.title('Manual Workload Reduction: XGBoost vs AutoGluon')
        plt.ylabel('Number of Claims')
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"[#] Saved workload visualization to '{output_path}'")
    else:
        print("Missing 'XGB_Action' or 'AG_Action' columns. Cannot generate plot.")


# ==========================================
# 2. LANGUAGE VERIFICATION EVALUATION
# ==========================================
def evaluate_language(
    report_path=PROJECT_ROOT / "data" / "language_evaluation_report.csv",
    output_path=PROJECT_ROOT / "data" / "language_accuracy_breakdown.png"
):
    print("\n" + "="*65)
    print("           LANGUAGE VERIFICATION PERFORMANCE REPORT")
    print("="*65)
    
    try:
        df = pd.read_csv(report_path)
    except FileNotFoundError:
        print(f"File not found: {report_path}. Skipping language evaluation.")
        return

    # Identify and filter out bot-flagged rows
    bot_condition = df['Lang_Reason'].astype(str).str.contains(r'bot|Sign in', case=False, na=False)
    skipped_count = bot_condition.sum()
    clean_df = df[~bot_condition].copy()
    
    if clean_df.empty:
        print(f"All rows ({skipped_count}) were skipped due to bot flagging. No data to analyze.")
        return

    # Treat Lang_Is_Match as a boolean
    clean_df['Lang_Is_Match'] = clean_df['Lang_Is_Match'].astype(str).str.strip().str.lower() == 'true'
    overall_accuracy = clean_df['Lang_Is_Match'].mean() * 100

    print(f"Total Rows in File      : {len(df)}")
    print(f"Skipped (Bot Flagged)   : {skipped_count}")
    print(f"Successfully Evaluated  : {len(clean_df)}")
    print(f"Overall System Accuracy : {overall_accuracy:.2f}%")
    print("=" * 65)

    lang_col = 'Expected Language'
    
    if lang_col in clean_df.columns:
        # Drop rows where Expected Language is completely empty so it doesn't skew the table
        clean_df = clean_df.dropna(subset=[lang_col])
        clean_df = clean_df[clean_df[lang_col].astype(str).str.strip() != ""]

        lang_stats = clean_df.groupby(lang_col).agg(
            Total_Cases=('Lang_Is_Match', 'count'),
            Correct_Matches=('Lang_Is_Match', 'sum')
        ).reset_index()
        
        lang_stats['Accuracy'] = (lang_stats['Correct_Matches'] / lang_stats['Total_Cases']) * 100
        lang_stats = lang_stats.sort_values(by='Total_Cases', ascending=False)

        print(f"{'Language/ID':<25} | {'Evaluated':<12} | {'Accuracy (%)':<12}")
        print("-" * 65)
        for _, row in lang_stats.iterrows():
            print(f"{str(row[lang_col]):<25} | {int(row['Total_Cases']):<12} | {row['Accuracy']:.2f}%")
        print("=" * 65)
        
        # Generate the Visualization
        if not lang_stats.empty:
            plt.figure(figsize=(12, max(6, len(lang_stats) * 0.4)))
            sns.set_theme(style="whitegrid")
            
            ax = sns.barplot(
                data=lang_stats,
                x='Accuracy',
                y=lang_col,
                palette='viridis',
                hue=lang_col,
                legend=False
            )
            
            plt.axvline(x=overall_accuracy, color='red', linestyle='--', alpha=0.7, label=f'Overall Accuracy ({overall_accuracy:.1f}%)')
            
            for i, p in enumerate(ax.patches):
                if i < len(lang_stats):
                    cases = int(lang_stats.iloc[i]['Total_Cases'])
                    ax.annotate(f" n={cases}", (p.get_width(), p.get_y() + p.get_height()/2), 
                                va='center', ha='left', fontsize=9, color='black')

            plt.title(f'Language Verification Accuracy Breakdown\n(Excluding {skipped_count} Bot-Flagged Claims)', fontsize=14, pad=15)
            plt.xlabel('Accuracy Percentage (%)')
            plt.ylabel('Target Language')
            plt.xlim(0, 105)
            plt.legend(loc='lower left')
            plt.tight_layout()
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300)
            print(f"[#] Saved language visualization to '{output_path}'\n")
        else:
            print("[!] No valid language data available to plot.")
    else:
        print(f"[!] Target column '{lang_col}' not found.")

if __name__ == "__main__":
    evaluate_models()
    evaluate_language()
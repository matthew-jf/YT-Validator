# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "matplotlib", "seaborn", "scikit-learn", "numpy", "tabulate"]
# ///

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, fbeta_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================================
# 1. MODEL EVALUATION (AUTOGLUON ONLY)
# ==========================================
def evaluate_models(
    preds_path=PROJECT_ROOT / "data" / "output_claims.csv",
    truth_path=PROJECT_ROOT / "data" / "unprocessed_claims_Matt_MCN_MARCH1Rev.csv",
    output_path=PROJECT_ROOT / "data" / "workload_reduction_ag.png",
):
    print("\n" + "="*65)
    print("           AUTOGLUON PERFORMANCE & WORKLOAD REDUCTION")
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

    # Calculate for AutoGluon
    ag_rate, ag_acc, ag_fns = calc_metrics('AG_Action')
    
    # Generate Original Text Report
    print(f"AutoGluon | Automation: {ag_rate:.2f}% | Accuracy: {ag_acc:.2f}% | False Negatives: {ag_fns}")
    print("="*65)

    # ---------------------------------------------------------
    # ORIGINAL CHART: Workload Reduction Chart
    # ---------------------------------------------------------
    if 'AG_Action' in valid_cases.columns:
        plt.figure(figsize=(8, 6))
        
        ax = sns.countplot(
            data=valid_cases, 
            x='AG_Action', 
            palette='muted', 
            order=['Auto Yes', 'Human Review', 'Auto No']
        )
        
        for container in ax.containers:
            ax.bar_label(container, padding=3, fontsize=11)
            
        plt.title('Manual Workload Reduction: AutoGluon', fontsize=14, pad=12)
        plt.xlabel('AI Recommended Action', fontsize=11, labelpad=8)
        plt.ylabel('Number of Claims', fontsize=11, labelpad=8)
        plt.tight_layout()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"[#] Saved workload visualization to '{output_path.name}'")
    else:
        print("Missing 'AG_Action' column. Cannot generate plot.")

    # ---------------------------------------------------------
    # NEW: Advanced Risk & Operational Scorecard Calculations
    # ---------------------------------------------------------
    print("\n" + "="*75)
    print(" AUTOGLUON ADVANCED PERFORMANCE REPORT")
    print("="*75)
    metrics_summary = []

    action_col = 'AG_Action'
    if action_col in valid_cases.columns:
            
        auto_subset = valid_cases[valid_cases[action_col] != 'Human Review'].copy()

        if len(auto_subset) > 0:
            y_true = auto_subset['verdict'].map({'Y': 1, 'N': 0}).astype(int)
            y_pred = auto_subset[action_col].map({'Auto Yes': 1, 'Auto No': 0}).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            automation_rate = (len(auto_subset) / len(valid_cases)) * 100
            accuracy = ((tp + tn) / len(auto_subset)) * 100
            recall = recall_score(y_true, y_pred, zero_division=0) * 100
            npv = (tn / (tn + fn)) * 100 if (tn + fn) > 0 else 0
            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0) * 100

            metrics_summary.append({
                'Model': 'AutoGluon',
                'Automation Rate': f"{automation_rate:.2f}%",
                'Automated Accuracy': f"{accuracy:.2f}%",
                'Recall (Safety)': f"{recall:.2f}%",
                'NPV (Auto-No Purity)': f"{npv:.2f}%",
                'F2-Score (Weighted)': f"{f2:.2f}%",
                'False Negatives': fn
            })

    if metrics_summary:
        scores_df = pd.DataFrame(metrics_summary)
        print(scores_df.to_markdown(index=False))
        print("\n* Recall (Safety): Higher means fewer valid claims were accidentally rejected.")
        print("* NPV (Auto-No Purity): Higher means your automated rejection bucket is cleaner.")
        print("* F2-Score: Weighted metric penalizing False Negatives twice as harshly as False Positives.")
        print("="*75)

    # ---------------------------------------------------------
    # CHART 1: Confusion Matrix Heatmap
    # ---------------------------------------------------------
    if 'AG_Action' in valid_cases.columns:
        action_order = ['Auto Yes', 'Auto No', 'Human Review']
        verdict_order = ['Y', 'N']

        crosstab_ag = pd.crosstab(valid_cases['verdict'], valid_cases['AG_Action'])
        crosstab_ag = crosstab_ag.reindex(index=verdict_order, columns=action_order, fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            crosstab_ag, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={'size': 16}, ax=ax
        )
        
        ax.set_title('AutoGluon: Prediction vs. Actual', fontsize=14, pad=12, weight='bold')
        ax.set_xlabel('AI Recommended Action', fontsize=11, labelpad=8)
        ax.set_ylabel('Actual Verdict (Historical)', fontsize=11, labelpad=8)

        plt.tight_layout()
        cm_output = output_path.parent / "confusion_matrix_ag.png"
        plt.savefig(cm_output, bbox_inches='tight')
        plt.close()
        print(f"[#] Saved confusion matrix to '{cm_output.name}'")

        # ---------------------------------------------------------
        # CHART 2: Operational Bar Chart
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))

        ax_ag = sns.countplot(
            data=valid_cases, x='AG_Action', hue='verdict',
            palette={'Y': '#2ecc71', 'N': '#e74c3c'},
            order=action_order, hue_order=verdict_order, ax=ax
        )
        
        for container in ax_ag.containers:
            ax_ag.bar_label(container, padding=3, fontsize=11)

        ax.set_title('AutoGluon Volume Breakdown', fontsize=14, pad=12, weight='bold')
        ax.set_xlabel('AI Recommended Action', fontsize=11, labelpad=8)
        ax.set_ylabel('Number of Claims', fontsize=11, labelpad=8)
        ax.legend(title='Actual Verdict', loc='upper right')

        plt.tight_layout()
        op_output = output_path.parent / "operational_breakdown_ag.png"
        plt.savefig(op_output, bbox_inches='tight')
        plt.close()
        print(f"[#] Saved operational breakdown to '{op_output.name}'")


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
            print(f"[#] Saved language visualization to '{output_path.name}'\n")
            plt.close()
        else:
            print("[!] No valid language data available to plot.")
    else:
        print(f"[!] Target column '{lang_col}' not found.")

if __name__ == "__main__":
    evaluate_models()
    evaluate_language()
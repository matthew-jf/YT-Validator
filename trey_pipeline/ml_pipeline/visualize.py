# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["pandas", "matplotlib", "seaborn", "numpy"]
# ///

from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def evaluate_and_visualize(
    data_path=PROJECT_ROOT / "data" / "output_claims.csv",
    output_path=PROJECT_ROOT / "data" / "business_value_dashboard.png",
):
    print(f"Loading inference results from {data_path}...")
    df = pd.read_csv(data_path)
    
    if df.empty:
        print("Data is empty. Skipping visualization.")
        return

    # ---------------------------------------------------------
    # 1. Console Reporting (The New Narrative)
    # ---------------------------------------------------------
    def get_automation_stats(action_col):
        total = len(df)
        auto_yes = (df[action_col] == 'Auto Yes').sum()
        auto_no = (df[action_col] == 'Auto No').sum()
        review = (df[action_col] == 'Human Review').sum()
        automated = auto_yes + auto_no
        return (automated/total)*100, (review/total)*100

    xgb_auto, xgb_rev = get_automation_stats('XGB_Action')
    ag_auto, ag_rev = get_automation_stats('AG_Action')

    print("\n" + "="*60)
    print(" BUSINESS VALUE & WORKLOAD REPORT")
    print("="*60)
    print(f"Total Claims Processed: {len(df):,}")
    print(f"XGBoost   | Automated: {xgb_auto:.1f}% | Sent to Human: {xgb_rev:.1f}%")
    print(f"AutoGluon | Automated: {ag_auto:.1f}% | Sent to Human: {ag_rev:.1f}%")
    print("="*60 + "\n")

    # ---------------------------------------------------------
    # 2. Visualizations Setup
    # ---------------------------------------------------------
    sns.set_theme(style="whitegrid")
    # Create a wide 3-panel dashboard
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Custom color palette for actions
    action_colors = {'Auto Yes': '#2ecc71', 'Auto No': '#e74c3c', 'Human Review': '#f1c40f'}

    # --- PANEL 1: The Action Funnel (Stacked Bar) ---
    # Prepare data for stacked bar
    xgb_counts = df['XGB_Action'].value_counts(normalize=True) * 100
    ag_counts = df['AG_Action'].value_counts(normalize=True) * 100
    
    plot_df = pd.DataFrame({'XGBoost': xgb_counts, 'AutoGluon': ag_counts}).T
    plot_df = plot_df.reindex(columns=['Auto Yes', 'Auto No', 'Human Review']).fillna(0)
    
    plot_df.plot(kind='bar', stacked=True, color=[action_colors[c] for c in plot_df.columns], ax=axes[0], edgecolor='white')
    axes[0].set_title('Workflow Distribution (The Action Funnel)', fontsize=14, pad=15)
    axes[0].set_ylabel('Percentage of Total Claims (%)')
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(title='Action Taken', loc='upper right')

    # --- PANEL 2: Confidence Distribution (The Sanity Check) ---
    # We will plot the Challenger (AutoGluon) confidence
    if 'AG_Confidence' in df.columns:
        sns.histplot(data=df, x='AG_Confidence', bins=50, kde=False, ax=axes[1], color='#3498db')
        axes[1].set_title('Model Confidence (AutoGluon)', fontsize=14, pad=15)
        axes[1].set_xlabel('Probability of "Yes"')
        axes[1].set_ylabel('Volume of Claims')
        axes[1].set_xlim(0, 1)
        
        # Draw threshold lines (assuming 0.60 and 0.97 from your previous script)
        axes[1].axvline(x=0.60, color='red', linestyle='--', alpha=0.7, label='Lower Threshold (0.60)')
        axes[1].axvline(x=0.97, color='green', linestyle='--', alpha=0.7, label='Upper Threshold (0.97)')
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, 'AG_Confidence column missing', ha='center', va='center')

    # --- PANEL 3: Automation Stability Over Time ---
    # TODO: Replace 'mock_date' with your actual timestamp column name
    # Synthesizing mock dates here just so the code runs out of the box
    np.random.seed(42)
    df['7/2/26'] = pd.to_datetime('today') - pd.to_timedelta(np.random.randint(0, 30, len(df)), unit='D')
    
    # Calculate daily automation rate for AutoGluon
    daily_stats = df.groupby(df['7/2/26'].dt.date).apply(
        lambda x: ((x['AG_Action'] != 'Human Review').sum() / len(x)) * 100
    ).reset_index(name='Automation_Rate')
    daily_stats.columns = ['Date', 'Automation Rate (%)']
    
    sns.lineplot(data=daily_stats, x='Date', y='Automation Rate (%)', marker='o', color='#9b59b6', ax=axes[2])
    axes[2].set_title('Automation Stability Over Time', fontsize=14, pad=15)
    axes[2].set_ylabel('Automated (%)')
    axes[2].set_ylim(0, 100)
    
    # Format x-axis dates nicely
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[2].tick_params(axis='x', rotation=45)

    # ---------------------------------------------------------
    # 3. Final Polish and Save
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved successfully to '{output_path}'")

if __name__ == "__main__":
    evaluate_and_visualize()
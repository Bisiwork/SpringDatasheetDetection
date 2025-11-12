"""
compare_experiments.py - Compare aggregated metrics between exp1 and exp2.

Loads agg_metrics.csv from exp1 and exp2, merges on 'model',
computes differences (exp2 - exp1) for numeric columns,
saves to output/experiment_differences.csv.
"""

import pandas as pd
from pathlib import Path

# Paths
EXP1_REPORTS = Path("output/exp0/reports")
EXP2_REPORTS = Path("output/exp1/reports")
EXP3_REPORTS = Path("output/exp2/reports")
OUTPUT_DIR = Path("output/comparision")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    # Load CSVs
    df1 = pd.read_csv(EXP1_REPORTS / "agg_metrics.csv")
    df2 = pd.read_csv(EXP2_REPORTS / "agg_metrics.csv")
    df3 = pd.read_csv(EXP3_REPORTS / "agg_metrics.csv")

    # Merge exp0 and exp1
    merged01 = pd.merge(df1, df2, on='model', suffixes=('_exp0', '_exp1'))

    # Merge with exp2
    merged = pd.merge(merged01, df3, on='model', suffixes=('', '_exp2'))

    # Rename columns for exp2
    merged = merged.rename(columns={
        'files': 'files_exp2',
        'fields_total': 'fields_total_exp2',
        'fields_correct': 'fields_correct_exp2',
        'accuracy': 'accuracy_exp2',
        'cost_usd': 'cost_usd_exp2'
    })

    # Numeric columns to compute differences (only those present in both)
    common_cols = ['files', 'fields_total', 'fields_correct', 'accuracy', 'cost_usd']

    for col in common_cols:
        col_exp1 = f"{col}_exp1"
        col_exp2 = f"{col}_exp2"
        diff_col = f"{col}_diff"
        merged[diff_col] = merged[col_exp2] - merged[col_exp1]

    # Save to CSV
    out_path = OUTPUT_DIR / "experiment_differences.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved differences to {out_path}")

    # --- Generate comparison plots from experiment_differences.csv ---
    import matplotlib.pyplot as plt
    import numpy as np
    # Read the differences file (to ensure columns are as expected)
    diff_df = pd.read_csv(out_path)

    # Plot settings
    plt.style.use('seaborn-v0_8')
    models = diff_df['model']

    # 1. Bar plot: accuracy_exp0 vs accuracy_exp1 vs accuracy_exp2
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width, diff_df['accuracy_exp0'], width, label='Exp0')
    ax.bar(x, diff_df['accuracy_exp1'], width, label='Exp1')
    ax.bar(x + width, diff_df['accuracy_exp2'], width, label='Exp2')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Model Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png')
    plt.close()

    # 2. Bar plot: cost_usd_exp0 vs cost_usd_exp1 vs cost_usd_exp2
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width, diff_df['cost_usd_exp0'], width, label='Exp0')
    ax.bar(x, diff_df['cost_usd_exp1'], width, label='Exp1')
    ax.bar(x + width, diff_df['cost_usd_exp2'], width, label='Exp2')
    ax.set_ylabel('Cost (USD)')
    ax.set_title('Cost by Model Across Experiments')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cost_comparison.png')
    plt.close()

    # 3. Bar plot: accuracy_diff (exp2 - exp1)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(models, diff_df['accuracy_diff'], color='tab:green')
    ax.set_ylabel('Accuracy Difference (Exp2 - Exp1)')
    ax.set_title('Accuracy Difference by Model (Exp2 - Exp1)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_difference.png')
    plt.close()

    # 4. Bar plot: cost_usd_diff (exp2 - exp1)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(models, diff_df['cost_usd_diff'], color='tab:orange')
    ax.set_ylabel('Cost Difference (USD, Exp2 - Exp1)')
    ax.set_title('Cost Difference by Model (Exp2 - Exp1)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cost_difference.png')
    plt.close()

    print(f"Saved comparison plots to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

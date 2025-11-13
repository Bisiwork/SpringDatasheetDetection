# Model Performance Evaluation

This module provides comprehensive evaluation tools for assessing the performance of different data extraction models.

## Features

- **Accuracy Evaluation**: Compare model predictions against ground truth data field-by-field
- **Cost Analysis**: Calculate total costs from usage CSV files
- **Comprehensive Metrics**: Overall accuracy, per-file accuracy, standard deviation, cost analysis
- **Visualizations**: 
  - Cost vs Accuracy scatter plots
  - Pareto frontier analysis for optimal trade-offs
  - Accuracy comparison bar charts
- **Flexible CLI**: Command-line interface for customizable evaluations

## Usage

### Basic Evaluation
```bash
python src/evaluate_tradeoff.py
```

### Evaluate Specific Models
```bash
python src/evaluate_tradeoff.py --models gpt4o gpt-5-mini
```

### Custom Configuration
```bash
python src/evaluate_tradeoff.py \
    --ground-truth-dir ground_truth \
    --results-dir . \
    --reports-dir reports \
    --tolerance 0.01
```

## Directory Structure

```
├── ground_truth/           # Ground truth JSON files
│   ├── Image.json
│   └── Image-2.json
├── results_<model>/        # Model prediction results
│   ├── *.json             # Prediction files
│   └── usage/             # Cost tracking CSV files
└── reports/               # Generated evaluation reports
    ├── agg_metrics.csv    # Aggregate metrics
    ├── cost_vs_accuracy.png
    ├── pareto_frontier.png
    └── accuracy_comparison.png
```

## Output Metrics

- **total_files**: Number of files processed
- **total_fields**: Total number of fields evaluated
- **correct_fields**: Number of correctly predicted fields
- **overall_accuracy**: Ratio of correct to total fields
- **mean_file_accuracy**: Average per-file accuracy
- **std_file_accuracy**: Standard deviation of file accuracies
- **total_cost_usd**: Total cost in USD from usage files

## Evaluation Results

Based on the current evaluation:

| Model | Accuracy | Cost ($) | Value (Acc/Cost) |
|-------|----------|----------|------------------|
| gpt4o | 0.621 | 0.0682 | 9.1 |
| gpt5_TimeOpt | 0.138 | 0.0584 | 2.4 |
| gpt5_noOpt | 0.172 | 0.3394 | 0.5 |
| gpt-5-mini | 0.207 | 0.1377 | 1.5 |

**Key Findings:**
- **Best Accuracy**: gpt4o (62.1%)
- **Lowest Cost**: gpt5_TimeOpt ($0.0584)
- **Best Value**: gpt4o (9.1 accuracy points per dollar)
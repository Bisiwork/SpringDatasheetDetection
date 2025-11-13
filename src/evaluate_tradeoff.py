#!/usr/bin/env python3
"""
evaluate_tradeoff.py - Performance evaluation for data extraction models

This script evaluates the performance of various models (e.g., GPT-4o, GPT-5-mini) 
in extracting data from spring-related documents by comparing model predictions 
against ground truth data and analyzing the trade-off between accuracy and cost.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import glob


class ModelEvaluator:
    """Evaluates model performance for data extraction tasks."""
    
    def __init__(self, 
                 ground_truth_dir: str = 'ground_truth',
                 results_dir: str = '.',
                 reports_dir: str = 'reports',
                 num_tolerance: float = 0.01):
        """
        Initialize the evaluator.
        
        Args:
            ground_truth_dir: Directory containing ground truth JSON files
            results_dir: Directory containing model result folders
            reports_dir: Directory to save evaluation reports
            num_tolerance: Tolerance for numerical field comparisons
        """
        self.ground_truth_dir = Path(ground_truth_dir)
        self.results_dir = Path(results_dir)
        self.reports_dir = Path(reports_dir)
        self.num_tolerance = num_tolerance
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
    
    def load_ground_truth(self) -> Dict[str, Dict[str, Any]]:
        """Load all ground truth files."""
        ground_truth = {}
        
        for file_path in self.ground_truth_dir.glob('*.json'):
            with open(file_path, 'r') as f:
                ground_truth[file_path.stem] = json.load(f)
        
        return ground_truth
    
    def load_model_predictions(self, model_name: str) -> Dict[str, Dict[str, Any]]:
        """Load predictions for a specific model."""
        predictions = {}
        
        # Look for model directory (e.g., results_gpt4o)
        model_dir = self.results_dir / f'results_{model_name.replace("-", "")}'
        if not model_dir.exists():
            model_dir = self.results_dir / f'{model_name}'
        
        if not model_dir.exists():
            print(f"Warning: Model directory not found for {model_name}")
            return predictions
        
        # Load JSON files from model directory
        for file_path in model_dir.glob('*.json'):
            with open(file_path, 'r') as f:
                predictions[file_path.stem] = json.load(f)
        
        return predictions
    
    def load_model_costs(self, model_name: str) -> float:
        """Load total cost for a model from usage CSV files."""
        total_cost = 0.0
        
        # Look for usage directory
        model_dir = self.results_dir / f'results_{model_name.replace("-", "")}'
        if not model_dir.exists():
            model_dir = self.results_dir / f'{model_name}'
        
        usage_dir = model_dir / 'usage'
        if not usage_dir.exists():
            return 0.0
        
        # Sum up costs from all CSV files
        for csv_file in usage_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                if 'USD' in df.columns:
                    # Look for TOTAL row first
                    if 'TOTAL' in df['stage'].values:
                        total_row = df[df['stage'] == 'TOTAL']
                        if not total_row.empty and pd.notna(total_row['USD'].iloc[0]):
                            total_cost += float(total_row['USD'].iloc[0])
                            continue
                    
                    # If no TOTAL row or TOTAL is NaN, sum individual costs
                    cost_df = df[df['stage'] != 'TOTAL'] if 'TOTAL' in df['stage'].values else df
                    # Handle numeric values, skip non-numeric entries
                    numeric_costs = pd.to_numeric(cost_df['USD'], errors='coerce')
                    valid_costs = numeric_costs.dropna()
                    total_cost += valid_costs.sum()
            except Exception as e:
                print(f"Warning: Could not read cost data from {csv_file}: {e}")
        
        return total_cost
    
    def compare_fields(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Tuple[int, int]:
        """
        Compare prediction with ground truth field by field.
        
        Returns:
            Tuple of (total_fields, correct_fields)
        """
        total_fields = 0
        correct_fields = 0
        
        for field, true_value in ground_truth.items():
            total_fields += 1
            pred_value = prediction.get(field)
            
            if pred_value is None:
                continue
            
            # Handle different data types
            if isinstance(true_value, (int, float)) and isinstance(pred_value, (int, float)):
                # Numerical comparison with tolerance
                if abs(float(pred_value) - float(true_value)) <= self.num_tolerance:
                    correct_fields += 1
            elif isinstance(true_value, str) and isinstance(pred_value, str):
                # String comparison (case-insensitive)
                if true_value.lower() == pred_value.lower():
                    correct_fields += 1
            else:
                # Direct comparison for other types
                if pred_value == true_value:
                    correct_fields += 1
        
        return total_fields, correct_fields
    
    def evaluate_model(self, model_name: str, ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a single model's performance."""
        predictions = self.load_model_predictions(model_name)
        cost = self.load_model_costs(model_name)
        
        total_files = 0
        total_fields_all = 0
        correct_fields_all = 0
        file_accuracies = []
        
        for file_name, true_data in ground_truth.items():
            if file_name in predictions:
                total_files += 1
                pred_data = predictions[file_name]
                
                total_fields, correct_fields = self.compare_fields(pred_data, true_data)
                total_fields_all += total_fields
                correct_fields_all += correct_fields
                
                # Calculate per-file accuracy
                file_accuracy = correct_fields / total_fields if total_fields > 0 else 0.0
                file_accuracies.append(file_accuracy)
        
        # Calculate overall accuracy
        overall_accuracy = correct_fields_all / total_fields_all if total_fields_all > 0 else 0.0
        
        return {
            'model': model_name,
            'total_files': total_files,
            'total_fields': total_fields_all,
            'correct_fields': correct_fields_all,
            'overall_accuracy': overall_accuracy,
            'mean_file_accuracy': np.mean(file_accuracies) if file_accuracies else 0.0,
            'std_file_accuracy': np.std(file_accuracies) if file_accuracies else 0.0,
            'total_cost_usd': cost
        }
    
    def find_available_models(self) -> List[str]:
        """Find all available model result directories."""
        models = []
        
        # Look for directories starting with 'results_'
        for path in self.results_dir.glob('results_*'):
            if path.is_dir():
                model_name = path.name.replace('results_', '')
                # Convert back to standard naming (e.g., gpt4o -> gpt-4o)
                if model_name.startswith('gpt') and model_name[3:].isdigit():
                    model_name = model_name[:3] + '-' + model_name[3:]
                elif model_name == 'gpt5mini':
                    model_name = 'gpt-5-mini'
                models.append(model_name)
        
        return models
    
    def generate_report(self, models: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate evaluation report for specified models."""
        if models is None:
            models = self.find_available_models()
        
        if not models:
            print("No models found for evaluation")
            return pd.DataFrame()
        
        ground_truth = self.load_ground_truth()
        if not ground_truth:
            print("No ground truth data found")
            return pd.DataFrame()
        
        print(f"Evaluating models: {models}")
        print(f"Ground truth files: {list(ground_truth.keys())}")
        
        results = []
        for model in models:
            print(f"Evaluating {model}...")
            result = self.evaluate_model(model, ground_truth)
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Save report
        report_path = self.reports_dir / 'agg_metrics.csv'
        df.to_csv(report_path, index=False)
        print(f"Report saved to: {report_path}")
        
        return df
    
    def plot_cost_vs_accuracy(self, df: pd.DataFrame) -> None:
        """Generate scatter plot of cost vs accuracy."""
        if df.empty:
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df['total_cost_usd'], df['overall_accuracy'], s=100, alpha=0.7)
        
        # Add model labels
        for idx, row in df.iterrows():
            plt.annotate(row['model'], 
                        (row['total_cost_usd'], row['overall_accuracy']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Total Cost (USD)')
        plt.ylabel('Overall Accuracy')
        plt.title('Model Performance: Cost vs Accuracy Trade-off')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.reports_dir / 'cost_vs_accuracy.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Cost vs Accuracy plot saved to: {plot_path}")
    
    def plot_pareto_frontier(self, df: pd.DataFrame) -> None:
        """Generate Pareto frontier plot."""
        if df.empty or len(df) < 2:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Sort by cost for Pareto frontier calculation
        df_sorted = df.sort_values('total_cost_usd')
        
        # Find Pareto optimal points (maximize accuracy, minimize cost)
        pareto_points = []
        max_accuracy_so_far = -1
        
        for idx, row in df_sorted.iterrows():
            if row['overall_accuracy'] > max_accuracy_so_far:
                pareto_points.append(idx)
                max_accuracy_so_far = row['overall_accuracy']
        
        # Plot all points
        plt.scatter(df['total_cost_usd'], df['overall_accuracy'], 
                   s=100, alpha=0.5, color='lightblue', label='All models')
        
        # Highlight Pareto optimal points
        pareto_df = df.loc[pareto_points]
        plt.scatter(pareto_df['total_cost_usd'], pareto_df['overall_accuracy'], 
                   s=150, alpha=0.8, color='red', label='Pareto optimal')
        
        # Draw Pareto frontier line
        pareto_df_sorted = pareto_df.sort_values('total_cost_usd')
        plt.plot(pareto_df_sorted['total_cost_usd'], pareto_df_sorted['overall_accuracy'], 
                'r--', alpha=0.6, linewidth=2)
        
        # Add labels
        for idx, row in df.iterrows():
            plt.annotate(row['model'], 
                        (row['total_cost_usd'], row['overall_accuracy']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Total Cost (USD)')
        plt.ylabel('Overall Accuracy')
        plt.title('Pareto Frontier: Optimal Cost-Accuracy Trade-offs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = self.reports_dir / 'pareto_frontier.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pareto frontier plot saved to: {plot_path}")
    
    def plot_accuracy_comparison(self, df: pd.DataFrame) -> None:
        """Generate bar chart comparing model accuracies."""
        if df.empty:
            return
        
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(df))
        bars = plt.bar(x_pos, df['overall_accuracy'], alpha=0.7)
        
        # Color bars based on accuracy
        colors = plt.cm.RdYlGn(df['overall_accuracy'])
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Models')
        plt.ylabel('Overall Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(x_pos, df['model'], rotation=45)
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for i, v in enumerate(df['overall_accuracy']):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save plot
        plot_path = self.reports_dir / 'accuracy_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy comparison plot saved to: {plot_path}")
    
    def run_full_evaluation(self, models: Optional[List[str]] = None) -> None:
        """Run complete evaluation pipeline."""
        print("Starting model performance evaluation...")
        
        # Generate report
        df = self.generate_report(models)
        
        if df.empty:
            print("No data to evaluate")
            return
        
        # Display summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(df.to_string(index=False))
        
        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_cost_vs_accuracy(df)
        self.plot_pareto_frontier(df)
        self.plot_accuracy_comparison(df)
        
        print(f"\nEvaluation complete! Reports and plots saved in: {self.reports_dir}")


def main():
    """Main function to run the evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model performance for data extraction')
    parser.add_argument('--models', nargs='*', help='Specific models to evaluate (e.g., gpt4o gpt-5-mini)')
    parser.add_argument('--ground-truth-dir', default='ground_truth', help='Ground truth directory')
    parser.add_argument('--results-dir', default='.', help='Results directory')
    parser.add_argument('--reports-dir', default='reports', help='Reports output directory')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Numerical tolerance for comparisons')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(
        ground_truth_dir=args.ground_truth_dir,
        results_dir=args.results_dir,
        reports_dir=args.reports_dir,
        num_tolerance=args.tolerance
    )
    
    evaluator.run_full_evaluation(models=args.models)


if __name__ == '__main__':
    main()
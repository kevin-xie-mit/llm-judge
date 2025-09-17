#!/usr/bin/env python3
"""
Judge Result Processor

This script processes the results from LLM-as-a-judge evaluations,
de-anonymizes model names, and generates comprehensive analysis reports.
"""

import os
import json
import re
import pandas as pd
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def parse_azure_batch_result(result_file_path: str) -> List[Dict[str, Any]]:
    """
    Parse Azure OpenAI batch result file.
    
    Args:
        result_file_path: Path to the batch result JSONL file
        
    Returns:
        List of parsed results
    """
    results = []
    
    with open(result_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                result = json.loads(line.strip())
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    return results


def extract_judge_evaluations(judge_response: str) -> List[Dict[str, Any]]:
    """
    Extract structured evaluations from judge response text.
    
    Args:
        judge_response: Raw response text from the judge
        
    Returns:
        List of extracted evaluation dictionaries
    """
    evaluations = []
    
    # Pattern to find model evaluations
    model_pattern = r'\*\*([^*]+)\*\*:\s*```json\s*(\{[^`]+\})\s*```'
    matches = re.findall(model_pattern, judge_response, re.DOTALL)
    
    for model_id, json_str in matches:
        try:
            evaluation = json.loads(json_str.strip())
            evaluation['model_id'] = model_id.strip()
            evaluations.append(evaluation)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {model_id}: {e}")
            continue
    
    # If no structured format found, try alternative parsing
    if not evaluations:
        # Try to find JSON blocks without model headers
        json_pattern = r'```json\s*(\{[^`]+\})\s*```'
        json_matches = re.findall(json_pattern, judge_response, re.DOTALL)
        
        for i, json_str in enumerate(json_matches):
            try:
                evaluation = json.loads(json_str.strip())
                evaluation['model_id'] = f"Model_{i+1}"  # Fallback ID
                evaluations.append(evaluation)
            except json.JSONDecodeError:
                continue
    
    return evaluations


def process_judge_results(batch_results: List[Dict[str, Any]], 
                         reference_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process judge results and create a structured DataFrame.
    
    Args:
        batch_results: Results from Azure batch API
        reference_data: Reference data with anonymous mappings
        
    Returns:
        DataFrame with processed results
    """
    processed_results = []
    
    # Create lookup for reference data
    reference_lookup = {ref['request_id']: ref for ref in reference_data}
    
    for result in batch_results:
        request_id = result['custom_id']
        
        if request_id not in reference_lookup:
            print(f"Warning: No reference data found for {request_id}")
            continue
            
        reference = reference_lookup[request_id]
        
        # Extract judge response
        try:
            judge_response = result['response']['body']['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            print(f"Error extracting response for {request_id}: {e}")
            continue
        
        # Parse judge evaluations
        evaluations = extract_judge_evaluations(judge_response)
        
        if not evaluations:
            print(f"Warning: No evaluations extracted from {request_id}")
            continue
        
        # Process each evaluation
        for evaluation in evaluations:
            # De-anonymize model name
            anonymous_id = evaluation['model_id']
            reverse_mapping = {v: k for k, v in reference['anonymous_mapping'].items()}
            original_model = reverse_mapping.get(anonymous_id, anonymous_id)
            
            # Find the corresponding prediction
            pred_info = None
            for pred in reference['predictions']:
                if pred['model_id'] == anonymous_id:
                    pred_info = pred
                    break
            
            if pred_info is None:
                print(f"Warning: No prediction info found for {anonymous_id} in {request_id}")
                continue
            
            processed_result = {
                'request_id': request_id,
                'task_name': reference['task_name'],
                'test_id': reference['test_id'],
                'model_name': original_model,
                'anonymous_id': anonymous_id,
                'prediction': pred_info['prediction'],
                'expected_output': reference['expected_output'],
                'hallucination': evaluation.get('hallucination', -1),
                'reason_hallucination': evaluation.get('reason_of_hallucination', ''),
                'omission': evaluation.get('omission', -1),
                'reason_omission': evaluation.get('reason_of_omission', ''),
                'incomplete': evaluation.get('incomplete', -1),
                'reason_incomplete': evaluation.get('reason_of_incomplete', ''),
                'instruction_following': evaluation.get('instruction_following', -1),
                'reason_instruction_following': evaluation.get('reason_of_instruction_following', ''),
                'judge_response': judge_response
            }
            
            processed_results.append(processed_result)
    
    return pd.DataFrame(processed_results)


def generate_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics from the evaluation results.
    
    Args:
        df: DataFrame with processed results
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {}
    
    # Overall statistics
    summary['total_evaluations'] = len(df)
    summary['unique_models'] = df['model_name'].nunique()
    summary['unique_tasks'] = df['task_name'].nunique()
    summary['unique_test_cases'] = df['test_id'].nunique()
    
    # Error rate statistics by metric
    metrics = ['hallucination', 'omission', 'incomplete', 'instruction_following']
    
    summary['overall_error_rates'] = {}
    for metric in metrics:
        valid_scores = df[df[metric] >= 0][metric]  # Exclude -1 (missing) values
        if len(valid_scores) > 0:
            summary['overall_error_rates'][metric] = {
                'error_rate': valid_scores.mean(),
                'total_cases': len(valid_scores),
                'error_cases': valid_scores.sum()
            }
    
    # Error rates by model
    summary['model_error_rates'] = {}
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        summary['model_error_rates'][model] = {}
        
        for metric in metrics:
            valid_scores = model_df[model_df[metric] >= 0][metric]
            if len(valid_scores) > 0:
                summary['model_error_rates'][model][metric] = {
                    'error_rate': valid_scores.mean(),
                    'total_cases': len(valid_scores),
                    'error_cases': valid_scores.sum()
                }
    
    # Error rates by task
    summary['task_error_rates'] = {}
    for task in df['task_name'].unique():
        task_df = df[df['task_name'] == task]
        summary['task_error_rates'][task] = {}
        
        for metric in metrics:
            valid_scores = task_df[task_df[metric] >= 0][metric]
            if len(valid_scores) > 0:
                summary['task_error_rates'][task][metric] = {
                    'error_rate': valid_scores.mean(),
                    'total_cases': len(valid_scores),
                    'error_cases': valid_scores.sum()
                }
    
    return summary


def create_visualizations(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualization plots for the evaluation results.
    
    Args:
        df: DataFrame with processed results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['hallucination', 'omission', 'incomplete', 'instruction_following']
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Overall error rates by metric
    fig, ax = plt.subplots(figsize=(10, 6))
    error_rates = []
    metric_names = []
    
    for metric in metrics:
        valid_scores = df[df[metric] >= 0][metric]
        if len(valid_scores) > 0:
            error_rates.append(valid_scores.mean())
            metric_names.append(metric.replace('_', ' ').title())
    
    bars = ax.bar(metric_names, error_rates)
    ax.set_ylabel('Error Rate')
    ax.set_title('Overall Error Rates by Evaluation Metric')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, error_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_error_rates.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error rates by model
    if df['model_name'].nunique() > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            model_error_rates = []
            model_names = []
            
            for model in sorted(df['model_name'].unique()):
                model_df = df[df['model_name'] == model]
                valid_scores = model_df[model_df[metric] >= 0][metric]
                if len(valid_scores) > 0:
                    model_error_rates.append(valid_scores.mean())
                    model_names.append(model)
            
            if model_error_rates:
                bars = axes[i].bar(range(len(model_names)), model_error_rates)
                axes[i].set_ylabel('Error Rate')
                axes[i].set_title(f'{metric.replace("_", " ").title()} by Model')
                axes[i].set_xticks(range(len(model_names)))
                axes[i].set_xticklabels(model_names, rotation=45, ha='right')
                axes[i].set_ylim(0, 1)
                
                # Add value labels
                for bar, rate in zip(bars, model_error_rates):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{rate:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_rates_by_model.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Heatmap of error rates by model and metric
    if df['model_name'].nunique() > 1:
        # Create pivot table
        heatmap_data = []
        models = sorted(df['model_name'].unique())
        
        for model in models:
            model_row = []
            model_df = df[df['model_name'] == model]
            for metric in metrics:
                valid_scores = model_df[model_df[metric] >= 0][metric]
                error_rate = valid_scores.mean() if len(valid_scores) > 0 else 0
                model_row.append(error_rate)
            heatmap_data.append(model_row)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))
        sns.heatmap(heatmap_data, 
                   xticklabels=[m.replace('_', ' ').title() for m in metrics],
                   yticklabels=models,
                   annot=True, fmt='.3f', cmap='Reds',
                   ax=ax)
        ax.set_title('Error Rate Heatmap: Models vs Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_rate_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()


def save_detailed_results(df: pd.DataFrame, output_path: str) -> None:
    """
    Save detailed results to CSV file.
    
    Args:
        df: DataFrame with processed results
        output_path: Path to save CSV file
    """
    # Create a copy for export (exclude long text fields for readability)
    export_df = df.copy()
    
    # Truncate long text fields
    text_fields = ['prediction', 'reason_hallucination', 'reason_omission', 
                   'reason_incomplete', 'reason_instruction_following']
    
    for field in text_fields:
        if field in export_df.columns:
            export_df[field] = export_df[field].str[:200] + '...'
    
    export_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Detailed results saved to {output_path}")


def main():
    """Main execution function."""
    # Configuration
    validation_dir = "/Users/kevinxie/Desktop/LLM CoT/validation"
    judge_dir = os.path.join(validation_dir, "judge_evaluation")
    
    # Input files
    batch_result_path = os.path.join(judge_dir, "judge_batch_results.jsonl")  # You'll get this from OpenAI
    reference_data_path = os.path.join(judge_dir, "judge_evaluation_reference.json")
    
    # Output files
    output_dir = os.path.join(judge_dir, "analysis")
    summary_path = os.path.join(output_dir, "summary_statistics.json")
    detailed_results_path = os.path.join(output_dir, "detailed_results.csv")
    plots_dir = os.path.join(output_dir, "plots")
    
    print("Starting judge result processing...")
    
    # Check if input files exist
    if not os.path.exists(batch_result_path):
        print(f"Error: Batch result file not found at {batch_result_path}")
        print("Please run the batch evaluation first and place the results file here.")
        return
    
    if not os.path.exists(reference_data_path):
        print(f"Error: Reference data file not found at {reference_data_path}")
        print("Please run judge_evaluation.py first to generate the reference data.")
        return
    
    # Load data
    print("Loading batch results...")
    batch_results = parse_azure_batch_result(batch_result_path)
    
    print("Loading reference data...")
    with open(reference_data_path, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    print(f"Loaded {len(batch_results)} batch results and {len(reference_data)} reference entries")
    
    # Process results
    print("Processing judge evaluations...")
    df = process_judge_results(batch_results, reference_data)
    
    if df.empty:
        print("Error: No valid evaluations were processed.")
        return
    
    print(f"Successfully processed {len(df)} evaluations")
    
    # Generate summary statistics
    print("Generating summary statistics...")
    summary = generate_summary_statistics(df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary statistics saved to {summary_path}")
    
    # Save detailed results
    save_detailed_results(df, detailed_results_path)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df, plots_dir)
    print(f"Plots saved to {plots_dir}")
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Unique models: {summary['unique_models']}")
    print(f"Unique tasks: {summary['unique_tasks']}")
    print(f"Unique test cases: {summary['unique_test_cases']}")
    
    print("\n=== Overall Error Rates ===")
    for metric, stats in summary['overall_error_rates'].items():
        print(f"{metric}: {stats['error_rate']:.3f} ({stats['error_cases']}/{stats['total_cases']})")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

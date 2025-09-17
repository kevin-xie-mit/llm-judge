#!/usr/bin/env python3
"""
Example Usage of LLM-as-a-Judge Mass Evaluation System

This script demonstrates how to use the evaluation system with a small subset of data.
"""

import os
import json
from judge_evaluation import (
    collect_all_predictions, 
    prepare_judge_data, 
    prepare_azure_batch_data_judge,
    save_batch_file,
    save_reference_data
)


def run_example_evaluation():
    """Run a small example evaluation to demonstrate the system."""
    
    print("=== LLM-as-a-Judge Example Usage ===\n")
    
    # Configuration for example
    validation_dir = "/Users/kevinxie/Desktop/LLM CoT/validation"
    output_dir = os.path.join(validation_dir, "example_evaluation")
    max_predictions_per_batch = 2  # Smaller for example
    
    print("Step 1: Collecting predictions from result files...")
    
    # Collect all predictions
    all_predictions = collect_all_predictions(validation_dir)
    
    if not all_predictions:
        print("No predictions found! Please check that result files exist.")
        return
    
    # Show summary
    total_tasks = len(all_predictions)
    total_models = sum(len(models) for models in all_predictions.values())
    print(f"Found: {total_tasks} tasks, {total_models} model-task combinations")
    
    # Show a sample of what was found
    print("\nSample of collected data:")
    for task_name, models_data in list(all_predictions.items())[:2]:  # Show first 2 tasks
        print(f"  Task: {task_name}")
        for model_name, predictions in list(models_data.items())[:3]:  # Show first 3 models
            print(f"    Model: {model_name} ({len(predictions)} predictions)")
    
    print(f"\nStep 2: Preparing judge evaluation data...")
    
    # Prepare judge data (limit to small subset for example)
    judge_requests = prepare_judge_data(all_predictions, max_predictions_per_batch)
    
    # Limit to first 10 requests for example
    judge_requests = judge_requests[:10]
    
    print(f"Created {len(judge_requests)} judge evaluation requests (limited for example)")
    
    print("\nStep 3: Creating Azure batch format...")
    
    # Create batch data
    batch_data = prepare_azure_batch_data_judge(
        judge_requests,
        model_name="gpt-4o",
        temperature=0.0,
        max_tokens=2000,  # Smaller for example
        seed=42
    )
    
    print(f"Generated {len(batch_data)} batch requests")
    
    print("\nStep 4: Saving example files...")
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    batch_file_path = os.path.join(output_dir, "example_batch.jsonl")
    reference_file_path = os.path.join(output_dir, "example_reference.json")
    
    save_batch_file(batch_data, batch_file_path)
    save_reference_data(judge_requests, reference_file_path)
    
    print(f"\nFiles saved:")
    print(f"  Batch file: {batch_file_path}")
    print(f"  Reference file: {reference_file_path}")
    
    print("\nStep 5: Showing example request structure...")
    
    # Show structure of first request
    if batch_data:
        example_request = batch_data[0]
        print("\nExample batch request structure:")
        print(f"  Custom ID: {example_request['custom_id']}")
        print(f"  Model: {example_request['body']['model']}")
        print(f"  Message length: {len(example_request['body']['messages'][1]['content'])} characters")
        
        # Show a snippet of the prompt
        prompt_snippet = example_request['body']['messages'][1]['content'][:500]
        print(f"\nPrompt snippet:")
        print(f"  {prompt_snippet}...")
    
    # Show reference data structure
    if judge_requests:
        example_ref = judge_requests[0]
        print(f"\nExample reference data structure:")
        print(f"  Request ID: {example_ref['request_id']}")
        print(f"  Task: {example_ref['task_name']}")
        print(f"  Test ID: {example_ref['test_id']}")
        print(f"  Number of predictions: {len(example_ref['predictions'])}")
        print(f"  Anonymous mapping: {example_ref['anonymous_mapping']}")
    
    print("\n=== Example Complete ===")
    print("\nNext steps:")
    print("1. Submit the batch file to OpenAI batch API")
    print("2. When results are ready, use judge_result_processor.py to analyze them")
    print("3. For full evaluation, run judge_evaluation.py without limits")
    
    # Estimate costs
    if batch_data:
        total_input_chars = sum(len(req['body']['messages'][1]['content']) for req in batch_data)
        estimated_input_tokens = total_input_chars // 4  # Rough estimate
        estimated_output_tokens = len(batch_data) * 1000  # Rough estimate
        estimated_cost = (estimated_input_tokens * 2.5 + estimated_output_tokens * 10) / 1000000
        
        print(f"\nCost estimate for example ({len(batch_data)} requests):")
        print(f"  Input tokens: ~{estimated_input_tokens:,}")
        print(f"  Output tokens: ~{estimated_output_tokens:,}")
        print(f"  Estimated cost: ${estimated_cost:.3f}")


def show_sample_judge_prompt():
    """Show what a judge prompt looks like."""
    
    print("\n=== Sample Judge Prompt ===")
    
    # Create a mock request to show prompt structure
    sample_request = {
        'task_name': 'Sample Task',
        'test_id': '12345',
        'instruction': 'Determine if the text mentions adverse drug effects.',
        'input': 'Patient developed rash after taking medication.',
        'expected_output': 'adverse drug effect: Yes',
        'predictions': [
            {
                'model_id': 'Model_abc123',
                'prediction': 'adverse drug effect: Yes'
            },
            {
                'model_id': 'Model_def456', 
                'prediction': 'adverse drug effect: No'
            }
        ]
    }
    
    from judge_evaluation import create_judge_prompt
    sample_prompt = create_judge_prompt(sample_request)
    
    # Show first 1000 characters
    print(sample_prompt[:1000])
    print("...")
    print(f"(Full prompt length: {len(sample_prompt)} characters)")


if __name__ == "__main__":
    print("LLM-as-a-Judge Mass Evaluation System - Example Usage\n")
    
    # Run the example
    run_example_evaluation()
    
    # Show sample prompt
    show_sample_judge_prompt()
    
    print("\n" + "="*50)
    print("Example completed successfully!")
    print("Check the 'example_evaluation' directory for output files.")

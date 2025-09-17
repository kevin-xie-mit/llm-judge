#!/usr/bin/env python3
"""
LLM-as-a-Judge Mass Evaluation System

This script prepares data for mass evaluation of LLM predictions using an LLM judge.
It processes result files from 4 representative models (Qwen2.5-72B-Instruct, GPT-4o, 
Llama-3.3-70B-Instruct, MeLLaMA-70B-chat), anonymizes model names, randomizes response 
order to avoid bias, and creates batch requests with multiple model responses per request
for efficient evaluation.

Key Features:
- Filters to only 4 representative models
- Combines multiple model responses into single evaluation requests
- Randomizes model response order per test case to avoid positional bias
- Anonymizes model names to prevent judge bias
- Creates batch-compatible requests for Azure OpenAI API
"""

import os
import json
import glob
import hashlib
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import random
from prompt import prompt


def collect_all_predictions(validation_dir: str) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Collect all model predictions from result files in the validation directory.
    Only includes the 4 representative models for evaluation.
    
    Args:
        validation_dir: Path to the validation directory containing task folders
        
    Returns:
        Dictionary organized as {task_name: {model_name: [predictions]}}
    """
    # Filter to only include these 4 representative models
    SELECTED_MODELS = {
        "Qwen2.5-72B-Instruct",
        "gpt-4o", 
        "Llama-3.3-70B-Instruct",
        "MeLLaMA-70B-chat"
    }
    
    all_predictions = defaultdict(lambda: defaultdict(list))
    
    # Find all result.json files
    pattern = os.path.join(validation_dir, "*", "*", "*.result.json")
    result_files = glob.glob(pattern)
    
    print(f"Found {len(result_files)} result files")
    
    for file_path in result_files:
        try:
            # Parse file path to extract task and model info
            path_parts = file_path.replace(validation_dir, "").strip("/").split("/")
            if len(path_parts) >= 3:
                task_name = path_parts[0]
                model_name = path_parts[1]
                
                # Skip models not in our selected set
                if model_name not in SELECTED_MODELS:
                    continue
                
                # Load predictions
                with open(file_path, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
                
                all_predictions[task_name][model_name].extend(predictions)
                print(f"Loaded {len(predictions)} predictions from {task_name}/{model_name}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return dict(all_predictions)


def anonymize_model_names(model_names: List[str], seed: int = 42) -> Dict[str, str]:
    """
    Create anonymous identifiers for model names to avoid self-preference bias.
    
    Args:
        model_names: List of model names to anonymize
        seed: Random seed for reproducible anonymization
        
    Returns:
        Dictionary mapping original model names to anonymous IDs
    """
    # Use deterministic hash-based approach for consistent anonymization
    anonymous_mapping = {}
    
    for model_name in sorted(model_names):  # Sort for consistency
        # Create a hash of the model name with seed
        hash_input = f"{model_name}_{seed}".encode('utf-8')
        hash_value = hashlib.md5(hash_input).hexdigest()[:8]
        anonymous_id = f"Model_{hash_value}"
        anonymous_mapping[model_name] = anonymous_id
    
    return anonymous_mapping


def prepare_judge_data(all_predictions: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Prepare data for LLM judge evaluation, creating one evaluation per test case with all model responses.
    
    Args:
        all_predictions: Collected predictions organized by task and model
        
    Returns:
        Dictionary organized as {task_name: [evaluation_requests]}
    """
    judge_requests_by_task = {}
    
    for task_name, models_data in all_predictions.items():
        print(f"\nProcessing task: {task_name}")
        
        # Get all model names for this task
        model_names = list(models_data.keys())
        if not model_names:
            continue
            
        # Create anonymous mapping for this task
        anonymous_mapping = anonymize_model_names(model_names)
        print(f"Anonymous mapping: {anonymous_mapping}")
        
        # Group predictions by test case ID
        test_cases = defaultdict(dict)
        for model_name, predictions in models_data.items():
            for pred in predictions:
                test_id = pred.get('id')
                if test_id is not None:
                    test_cases[test_id][model_name] = pred
        
        print(f"Found {len(test_cases)} unique test cases")
        
        # Initialize requests list for this task
        task_requests = []
        
        # Create one evaluation request per test case with all model responses
        for test_id, model_predictions in test_cases.items():
            # Skip test cases that don't have predictions from all models
            if len(model_predictions) < len(model_names):
                print(f"Skipping test case {test_id} - missing predictions from some models")
                continue
                
            # Get the base test case info (same across all models)
            base_pred = next(iter(model_predictions.values()))
            
            # Create anonymized model responses with randomized order
            model_responses = []
            model_names_list = list(model_predictions.keys())
            random.shuffle(model_names_list)  # Randomize order to avoid bias
            
            for model_name in model_names_list:
                pred_data = model_predictions[model_name]
                anonymous_id = anonymous_mapping[model_name]
                
                model_responses.append({
                    "model_id": anonymous_id,
                    "prediction": pred_data.get('pred', ''),
                    "original_model": model_name  # Keep for reference
                })
            
            judge_request = {
                "request_id": f"{task_name}_{test_id}",
                "task_name": task_name,
                "test_id": test_id,
                "instruction": base_pred.get('instruction', ''),
                "input": base_pred.get('input', ''),
                "expected_output": base_pred.get('output', ''),
                "model_responses": model_responses,
                "anonymous_mapping": anonymous_mapping,  # For later reference
                "randomized_order": [resp["original_model"] for resp in model_responses]  # Track order
            }
            
            task_requests.append(judge_request)
        
        judge_requests_by_task[task_name] = task_requests
        print(f"Created {len(task_requests)} evaluation requests for {task_name}")
    
    return judge_requests_by_task


def create_judge_user_content(request_data: Dict[str, Any]) -> str:
    """
    Create the user message content for the LLM judge. The system prompt is
    provided separately from prompt.py to be identical across all requests.

    Args:
        request_data: Evaluation request data with multiple model responses

    Returns:
        User message string containing the evaluation context and all model predictions
    """
    user_content = "## Evaluation Request\n\n"
    user_content += f"**Original Clinical Text:** {request_data['input']}\n\n"
    user_content += f"**Task Instructions:** {request_data['instruction']}\n\n"
    user_content += f"**Expected Output:** {request_data['expected_output']}\n\n"

    # Add all model predictions (anonymized, no model names)
    user_content += "**Model Predictions to Evaluate:**\n\n"
    for i, response in enumerate(request_data['model_responses'], 1):
        user_content += f"**{response['model_id']}:**\n"
        user_content += f"{response['prediction']}\n\n"

    # Clarify the expected response format (JSON object for each model as per system prompt)
    user_content += "Evaluate each model prediction and return a JSON object with scores for each model, "
    user_content += "strictly matching the JSON schema in the system prompt. "
    user_content += "Do not include any additional text outside of the JSON object.\n"

    return user_content


def prepare_azure_batch_data_judge(
    judge_requests: List[Dict[str, Any]],
    model_name: str = "gpt-4o",
    prompt_mode: str = "judge",
    split: str = "test",
    temperature: float = 0.0,
    top_p: float = 0.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_token_output: int = 4000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Prepare batch data compatible with Azure OpenAI API format, following the
    structure of prepare_azure_batch_data in azure.py.
    
    Args:
        judge_requests: List of judge evaluation requests
        model_name: OpenAI model to use for evaluation
        prompt_mode: Identifier for the prompt configuration (fixed to 'judge')
        split: Data split label to include in custom_id
        temperature: Sampling temperature
        top_p: Nucleus sampling value
        frequency_penalty: Frequency penalty
        presence_penalty: Presence penalty
        max_token_output: Maximum tokens for response
        seed: Random seed
        
    Returns:
        List of batch requests formatted for Azure OpenAI API
    """
    batch_data = []
    
    for request in judge_requests:
        # Build messages using a fixed system prompt and dynamic user content
        user_content = create_judge_user_content(request)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]
        
        # Construct custom_id for multi-model evaluation
        task_name = request.get("task_name", "TASK")
        test_id = request.get("test_id", "0")
        custom_id = f"{task_name}|{model_name}|{prompt_mode}|{split}|{test_id}|multi_model"

        # Format as Azure batch request
        batch_request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_token_output,
                "seed": seed,
            },
        }
        
        batch_data.append(batch_request)
    
    return batch_data


def save_batch_file(batch_data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save batch data to JSONL file format required by OpenAI batch API.
    
    Args:
        batch_data: List of batch requests
        output_path: Path to save the JSONL file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for request in batch_data:
            f.write(json.dumps(request, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(batch_data)} batch requests to {output_path}")


def save_task_batch_files(judge_requests_by_task: Dict[str, List[Dict[str, Any]]], output_dir: str) -> List[str]:
    """
    Save separate batch files for each task.
    
    Args:
        judge_requests_by_task: Dictionary of evaluation requests organized by task
        output_dir: Directory to save the batch files
        
    Returns:
        List of paths to the saved batch files
    """
    saved_files = []
    
    for task_name, task_requests in judge_requests_by_task.items():
        # Create batch data for this task
        batch_data = prepare_azure_batch_data_judge(task_requests)
        
        # Create safe filename from task name
        safe_task_name = task_name.replace("/", "_").replace(" ", "_")
        batch_file_path = os.path.join(output_dir, f"{safe_task_name}_judge_evaluation.jsonl")
        
        # Save the batch file
        save_batch_file(batch_data, batch_file_path)
        saved_files.append(batch_file_path)
    
    return saved_files


def save_reference_data(judge_requests_by_task: Dict[str, List[Dict[str, Any]]], output_path: str) -> None:
    """
    Save reference data for later analysis of judge results.
    
    Args:
        judge_requests_by_task: Original judge requests organized by task with anonymous mappings
        output_path: Path to save the reference data
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(judge_requests_by_task, f, ensure_ascii=False, indent=2)
    
    print(f"Saved reference data to {output_path}")


def main():
    """Main execution function."""
    # Configuration
    validation_dir = "/Users/kevinxie/Desktop/LLM CoT/validation"
    output_dir = os.path.join(validation_dir, "judge_evaluation")
    
    print("Starting LLM-as-a-Judge mass evaluation preparation...")
    
    # Step 1: Collect all predictions
    print("\n=== Step 1: Collecting Predictions ===")
    all_predictions = collect_all_predictions(validation_dir)
    
    total_tasks = len(all_predictions)
    total_models = sum(len(models) for models in all_predictions.values())
    total_predictions = sum(
        len(preds) for models in all_predictions.values() 
        for preds in models.values()
    )
    
    print(f"Summary: {total_tasks} tasks, {total_models} model-task combinations, {total_predictions} total predictions")
    
    # Step 2: Prepare judge evaluation data
    print("\n=== Step 2: Preparing Judge Evaluation Data ===")
    judge_requests_by_task = prepare_judge_data(all_predictions)
    
    total_requests = sum(len(requests) for requests in judge_requests_by_task.values())
    print(f"Created {total_requests} judge evaluation requests across {len(judge_requests_by_task)} tasks")
    
    # Step 3: Save separate batch files for each task
    print("\n=== Step 3: Saving Task-Specific Batch Files ===")
    saved_batch_files = save_task_batch_files(judge_requests_by_task, output_dir)
    
    # Step 4: Save reference data
    print("\n=== Step 4: Saving Reference Data ===")
    reference_file_path = os.path.join(output_dir, "judge_evaluation_reference.json")
    save_reference_data(judge_requests_by_task, reference_file_path)
    
    print(f"\n=== Evaluation Preparation Complete ===")
    print(f"Created {len(saved_batch_files)} task-specific batch files:")
    for file_path in saved_batch_files:
        print(f"  - {os.path.basename(file_path)}")
    print(f"Reference file: {reference_file_path}")
    print(f"Total requests: {total_requests}")
    
    # Estimate cost (calculate from first task as sample)
    if saved_batch_files:
        sample_requests = next(iter(judge_requests_by_task.values()))
        sample_batch_data = prepare_azure_batch_data_judge(sample_requests[:5])  # Sample first 5
        if sample_batch_data:
            avg_tokens_per_request = len(sample_batch_data[0]['body']['messages'][1]['content']) // 4
            estimated_input_tokens = total_requests * avg_tokens_per_request
            estimated_output_tokens = total_requests * 1000  # Rough estimate
            estimated_cost = (estimated_input_tokens * 2.5 + estimated_output_tokens * 10) / 1000000
            print(f"Estimated cost (GPT-4o): ${estimated_cost:.2f}")


if __name__ == "__main__":
    main()

    # Selected representative models (automatically filtered):
    # - Qwen2.5-72B-Instruct
    # - gpt-4o  
    # - Llama-3.3-70B-Instruct
    # - MeLLaMA-70B-chat
    #
    # Changes made:
    # 1. Only processes the 4 representative models above
    # 2. Creates one evaluation request per test case with all model responses
    # 3. Randomizes model response order per test case to avoid positional bias
    # 4. Updates prompt format to handle multiple model evaluations
    # 5. Anonymizes model names to prevent judge bias

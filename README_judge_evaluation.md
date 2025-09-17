# LLM-as-a-Judge Mass Evaluation System

This system provides a comprehensive solution for evaluating multiple LLM predictions using an LLM judge. It handles model anonymization, batch processing, and detailed analysis of results.

## Overview

The system consists of three main components:

1. **Data Preparation** (`judge_evaluation.py`): Collects predictions from multiple models and prepares batch evaluation requests
2. **Result Processing** (`judge_result_processor.py`): Processes judge evaluations and generates analysis reports
3. **Judge Prompt** (`prompt.py`): Contains the evaluation criteria and instructions for the LLM judge

## Key Features

- **Model Anonymization**: Prevents self-preference bias by anonymizing model names
- **Batch Processing**: Efficiently evaluates multiple predictions in single requests
- **Comprehensive Analysis**: Generates detailed statistics, visualizations, and reports
- **Azure OpenAI Compatible**: Works with existing Azure batch processing infrastructure

## Workflow

### Step 1: Prepare Evaluation Data

```bash
python judge_evaluation.py
```

This script will:
- Scan all result files in the validation directory
- Collect predictions from different models
- Anonymize model names using deterministic hashing
- Create batch requests for the judge
- Generate two output files:
  - `judge_evaluation/judge_evaluation_batch.jsonl`: Batch requests for OpenAI API
  - `judge_evaluation/judge_evaluation_reference.json`: Reference data for de-anonymization

### Step 2: Submit to OpenAI Batch API

Use the existing Azure infrastructure or OpenAI batch API to process the requests:

```python
# Example using existing azure.py infrastructure
from azure import create_azure_client, save_azure_batch_file

# The batch file is already in the correct format
batch_file = "judge_evaluation/judge_evaluation_batch.jsonl"

# Submit to Azure/OpenAI batch processing
# (Follow your existing batch submission process)
```

### Step 3: Process Results

Once you receive the batch results from OpenAI:

```bash
# Place the result file as: judge_evaluation/judge_batch_results.jsonl
python judge_result_processor.py
```

This will generate:
- Detailed analysis in `judge_evaluation/analysis/`
- Summary statistics (`summary_statistics.json`)
- Detailed results CSV (`detailed_results.csv`)
- Visualization plots in `plots/` directory

## Configuration

### Key Parameters in `judge_evaluation.py`:

```python
max_predictions_per_batch = 3  # Number of predictions per evaluation request
model_name = "gpt-4o"         # Judge model to use
temperature = 0.0             # Sampling temperature
max_tokens = 4000             # Maximum response tokens
seed = 42                     # Random seed for reproducibility
```

### Evaluation Metrics

The judge evaluates each prediction on four binary criteria:

1. **Hallucination**: Information not present in the original clinical text
2. **Omission**: Missing relevant clinical information
3. **Incomplete**: Cut-off or unfinished responses
4. **Instruction Following**: Adherence to task instructions and format

## File Structure

```
validation/
├── judge_evaluation/
│   ├── judge_evaluation_batch.jsonl      # Batch requests for OpenAI
│   ├── judge_evaluation_reference.json   # Reference data
│   ├── judge_batch_results.jsonl         # Results from OpenAI (you provide)
│   └── analysis/
│       ├── summary_statistics.json       # Overall statistics
│       ├── detailed_results.csv          # Full results table
│       └── plots/
│           ├── overall_error_rates.png   # Overall error rates
│           ├── error_rates_by_model.png  # Model comparison
│           └── error_rate_heatmap.png    # Model vs metric heatmap
├── judge_evaluation.py                   # Data preparation script
├── judge_result_processor.py             # Result analysis script
└── prompt.py                            # Judge evaluation prompt
```

## Example Output

### Summary Statistics

```json
{
  "total_evaluations": 1250,
  "unique_models": 15,
  "unique_tasks": 3,
  "overall_error_rates": {
    "hallucination": {
      "error_rate": 0.156,
      "total_cases": 1250,
      "error_cases": 195
    },
    "omission": {
      "error_rate": 0.234,
      "total_cases": 1250,
      "error_cases": 293
    }
  }
}
```

### Model Comparison

The system generates visualizations comparing:
- Overall error rates across all metrics
- Error rates by individual models
- Heatmap showing model performance across different evaluation criteria

## Customization

### Adding New Evaluation Criteria

1. Update the prompt in `prompt.py`
2. Modify the JSON schema in the prompt
3. Update parsing logic in `judge_result_processor.py`

### Changing Anonymization

Modify the `anonymize_model_names()` function in `judge_evaluation.py` to use different anonymization strategies.

### Batch Size Optimization

Adjust `max_predictions_per_batch` based on:
- Context window limits of your judge model
- Desired evaluation quality vs efficiency trade-off
- API rate limits and costs

## Cost Estimation

The system provides cost estimates based on:
- Input token count (prompt + predictions)
- Expected output tokens (evaluation responses)
- Current OpenAI pricing

For GPT-4o batch processing, typical costs range from $0.10-$0.50 per 1000 evaluations.

## Troubleshooting

### Common Issues

1. **No predictions found**: Check file paths and result file formats
2. **JSON parsing errors**: Verify judge response format matches expected structure
3. **Missing reference data**: Ensure you run data preparation before processing results
4. **Visualization errors**: Install required packages: `matplotlib`, `seaborn`, `pandas`

### Debug Mode

Add debug prints or modify logging levels in the scripts to troubleshoot specific issues.

## Dependencies

```bash
pip install pandas matplotlib seaborn numpy
```

The system also uses standard library modules: `json`, `glob`, `hashlib`, `re`, `os`, `collections`.

import os
import json
import time
import regex
import openai
import tiktoken
from openai import AzureOpenAI
from tqdm import tqdm
from dataset.process import get_formatted_chat
from dataset.config import transform_instruction_to_cot


def create_azure_client(azure_endpoint, api_key, api_version):
    """
    Create an instance of the AzureOpenAI class

    Input:
        azure_endpoint: string, the endpoint of the azure openai
        api_key: string, the api key of the azure openai
        api_version: string, the api version of the azure openai

    Output:
        azure_client: AzureOpenAI, the instance of the AzureOpenAI class

    """
    azure_client = AzureOpenAI(
        azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version
    )

    return azure_client


def azure_query_single(
    client,
    input,
    model_name="gpt-4-turbo",
    examples=[],
    temperature=0,
    max_tokens=100,
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    seed=42,
):
    """
    Query the Azure OpenAI model with a single data

    Input:
        dict_data: dictionary, including input, instruction, model name, and pred
        client: AzureOpenAI, the instance of the AzureOpenAI class
        model_name: string, the model name
        temperature: float, the temperature for the model
        max_tokens: int, the max_tokens for the model
        top_p: float, the top_p for the model
        frequency_penalty: float, the frequency_penalty for the model
        presence_penalty: float, the presence_penalty for the model

    Output:
        response_text: string, the response text
        response: AzureResponse, the response of the Azure model

    """
    # Construct message
    if isinstance(input, dict):
        messages = [
            {"role": "system", "content": input["instruction"]},
            {"role": "user", "content": input["input"]},
        ]
    else:
        messages = input

    # Add examples, if any
    for ex in examples:
        messages.append({"role": "user", "content": ex["input"]})
        messages.append({"role": "user", "content": ex["output"]})

    # Query the model
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        seed=seed,
    )

    # Get the output
    response_text = response.choices[0].message.content

    return response_text, response


def create_azure_batch_data(
    task_name,
    model_name,
    prompt_mode,
    split="test",
    path_dir_raw="dataset_raw",
    path_dir_batch="azure/input",
    temperature=0,
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    max_token_input=100 * 1024,
    max_token_output=3072,
    seed=42,
):
    """
    Create a list of dictionary for batch data

    Input:
        task_name: string, the task name
        model_name: string, the model name
        num_example: int, the number of example
        split: string, the split of the data

    Output:
        list_dict_data_batch: list of dictionary, including input, instruction, and model name

    """
    # Load data
    path_file_task = f"{path_dir_raw}/{task_name}.SFT.json"
    with open(path_file_task, "r", encoding="utf-8") as f:
        list_dict_data = json.load(f)
    # - Filter the data
    list_dict_data = [
        dict_data for dict_data in list_dict_data if dict_data["split"] == split
    ]
    print(f"Task: {task_name}")
    print(f" - {split.capitalize()} split: {len(list_dict_data)} samples")

    print(f" - Model: {model_name}")
    print(f" - Prompt: {prompt_mode}")

    # Prepare instruction
    if "direct" in prompt_mode:
        pass
    elif "cot" in prompt_mode:
        for dict_data in list_dict_data:
            dict_data["instruction"] = transform_instruction_to_cot(
                dict_data["instruction"]
            )
        print(f" - Transform the instruction to the CoT format")
        print(f" - Instruction: \n{dict_data['instruction']}")

    # Prepare example
    if "shot" in prompt_mode:
        num_example = int(regex.findall(r"\d+", prompt_mode)[0])
        path_file_example = f"{path_dir_raw}/example/{task_name}.example.json"
        with open(path_file_example, "r", encoding="utf-8") as f:
            list_dict_example = json.load(f)
        examples = list_dict_example[:num_example]
        print(f" - Prepare {num_example} examples")
    else:
        examples = []
        print(" - No example")

    # Get formatted data
    list_dict_data_batch = prepare_azure_batch_data(
        task_name=task_name,
        model_name=model_name,
        prompt_mode=prompt_mode,
        split=split,
        list_dict_data=list_dict_data,
        examples=examples,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_token_input=max_token_input,
        max_token_output=max_token_output,
        seed=seed,
    )

    # Save data
    path_dir_azure_batch = f"{path_dir_batch}/{model_name}/{prompt_mode}"
    if not os.path.exists(path_dir_azure_batch):
        os.makedirs(path_dir_azure_batch)
    path_file_azure_batch = f"{path_dir_azure_batch}/{task_name}.batch.jsonl"
    save_azure_batch_file(list_dict_data_batch, path_file_azure_batch)
    print(f" - Save {len(list_dict_data_batch)} to {path_file_azure_batch}")

    return list_dict_data_batch


def prepare_azure_batch_data(
    task_name,
    model_name,
    prompt_mode,
    split,
    list_dict_data,
    examples=[],
    temperature=0,
    top_p=0,
    frequency_penalty=0,
    presence_penalty=0,
    max_token_input=100 * 1024,
    max_token_output=3 * 1024,
    seed=42,
    num_token_reserve=5,
):
    """
    Prepare a list of dictionary for batch data

    Input:
        task_name: string, the task name
        prompt_mode: string, the prompt mode
        list_dict_data: list of dictionary, including input, instruction, and split
        temperature: float, the temperature for the model
        top_p: float, the top_p for the model
        frequency_penalty: float, the frequency_penalty for the model
        presence_penalty: float, the presence_penalty for the model
        max_token_outputs: int, the max_token_outputs for the model

    Output:
        list_dict_data_batch: list of dictionary, including input, instruction, and model name

    """
    if "gpt-4o" in model_name or "gpt-4o-mini" in model_name:
        # o200k_base: gpt-4o, gpt-4o-mini
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        print(" - Loading tokenizer of gpt-4o")
    else:
        # cl100k_base: gpt-4-turbo, gpt-4, gpt-3.5-turbo, text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
        tokenizer = tiktoken.encoding_for_model("gpt-35-turbo")
        print(" - Loading tokenizer of gpt-35-turbo")

    if "gpt-35" in model_name:
        max_token_input_valid = 16 * 1024 - max_token_output - num_token_reserve
        max_token_input = (
            max_token_input_valid
            if max_token_input > max_token_input_valid
            else max_token_input
        )

    print(f" - Max token input: {max_token_input}")
    print(f" - Max token output: {max_token_output}")

    language = list_dict_data[0]["language"]

    # Prepare id prefix
    exp_name = f"{task_name}|{model_name}|{prompt_mode}|{split}"

    list_dict_data_batch = []
    for idx, dict_data in enumerate(list_dict_data):
        example_ids, list_message = get_formatted_chat(
            tokenizer=tokenizer,
            language=language,
            input_system=dict_data["instruction"],
            input_user=dict_data["input"],
            max_token_input=max_token_input,
            times_word_token=4,
            num_token_reserve=5,
            examples=examples,
            flag_openai=True,
        )
        dict_data_batch = {
            "custom_id": f"{exp_name}|{dict_data['id']}",
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": model_name,
                "messages": list_message,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_token_output,
                "seed": seed,
            },
        }
        list_dict_data_batch.append(dict_data_batch)

    return list_dict_data_batch


def merge_azure_batch_data(
    model_name,
    prompt_mode,
    path_dir_raw="azure/input",
    path_dir_merged="azure/input/merged",
    max_lines=50000,
    max_size_mb=180,
):
    """
    Merge multiple Azure batch files into a single file, which is useful for reducing the number of files.
    - Each merged file will contain up to `max_lines` lines and be no larger than `max_size_mb` MB.
    - Each file (task) is not split across multiple output files.

    Args:
        model_name (str): The model name, which is part of the input directory structure.
        prompt_mode (str): The prompt mode directory name under `azure/input/{model}/`.
        max_lines (int): Maximum number of lines per merged file.
        max_size_mb (int): Maximum size of a merged file in megabytes.

    Returns:
        None
    """

    path_dir_raw = f"{path_dir_raw}/{model_name}/{prompt_mode}"

    # Get list of files and process task names
    list_path_file = [
        os.path.join(path_dir_raw, file)
        for file in os.listdir(path_dir_raw)
        if file.endswith(".jsonl")
    ]

    # Create the output directory if it doesn't exist
    os.makedirs(path_dir_merged, exist_ok=True)

    chunk_index = 0
    current_lines = []
    current_size = 0

    # Convert MB to bytes
    max_size_bytes = max_size_mb * 1024 * 1024

    for task_file in list_path_file:
        # 1. Read the file into memory
        with open(task_file, "r", encoding="utf-8") as infile:
            file_lines = [line.rstrip("\n") for line in infile]

        # 2. Calculate the size of the file in bytes
        file_size = sum(len(line.encode("utf-8")) for line in file_lines)

        # 3. Check if adding this file to the current chunk will exceed the limits
        #    If so, write out the current chunk and reset the state
        if (len(current_lines) + len(file_lines) > max_lines) or (
            current_size + file_size > max_size_bytes
        ):
            # Write out the current chunk
            output_file = os.path.join(
                path_dir_merged, f"{model_name}.{prompt_mode}.chunk_{chunk_index}.jsonl"
            )
            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.write("\n".join(current_lines))
            print(
                f"Created: {output_file} with {len(current_lines)} lines "
                f"and size {current_size / (1024 * 1024):.2f} MB"
            )

            # Reset the state
            chunk_index += 1
            current_lines = []
            current_size = 0

        # 4. Add the current file to the chunk
        current_lines.extend(file_lines)
        current_size += file_size

    # 5. Write out the last chunk
    if current_lines:
        output_file = os.path.join(
            path_dir_merged, f"{model_name}.{prompt_mode}.chunk_{chunk_index}.jsonl"
        )
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(current_lines))
        print(
            f"Created: {output_file} with {len(current_lines)} lines "
            f"and size {current_size / (1024 * 1024):.2f} MB"
        )


def parse_azure_batch_result(path_file_result_batch):
    """
    Extract the result from the openai batch file

    Input:
        path_file_result_batch: string, the path of the openai batch file

    Output:
        list_dict_id_result: list of dictionary, including id, output, error, token_input, and token_output

    """
    with open(path_file_result_batch, "r", encoding="utf-8") as f:
        list_dict_result = [json.loads(line) for line in f.readlines()]
    list_dict_id_result = []
    for idx_result, dict_result in enumerate(list_dict_result):
        try:
            dict_id_result = {
                "id": dict_result["custom_id"],
                "output": dict_result["response"]["body"]["choices"][0]["message"][
                    "content"
                ],
                "error": dict_result["error"],
            }
        except:
            print(f"Accident at {idx_result}")
            dict_id_result = {
                "id": dict_result["custom_id"],
                "output": "",
                "error": "",
            }
        list_dict_id_result.append(dict_id_result)

    return list_dict_id_result


def process_azure_result_to_task_result(
    path_dir_azure_result="azure/output",
    path_dir_task_save="result",
    path_dir_raw="dataset_raw",
    path_dir_batch="azure/input",
    decoding_strategy="greedy",
    seed=42,
):
    """
    Process the result from the azure batch files

    Input:
        list_path_file: list of string, the list of path file
        path_dir_result: string, the path of the result directory
        path_dir_save: string, the path of the save directory

    Output:
        None

    """
    # Get the list of path file, which is the result file of azure batch job
    list_path_file = [
        os.path.join(path_dir_azure_result, path_file)
        for path_file in os.listdir(path_dir_azure_result)
        if path_file.endswith(".jsonl")
    ]

    # Parse the result, as the result contains multiple tasks
    for path_file_result in list_path_file:

        # Read the result file
        list_dict_id_result = parse_azure_batch_result(path_file_result)

        # Split the whole result into different tasks
        dict_task_data_all = {}
        for dict_result in list_dict_id_result:
            # Get the task information: task, model_name, prompt_mode
            task, model_name, prompt_mode, split, id = dict_result["id"].split("|")

            if task not in dict_task_data_all:
                dict_task_data_all[task] = {
                    "model_name": model_name,
                    "prompt_mode": prompt_mode,
                    "split": split,
                    "list_dict_result": [],
                }

            dict_task_data_all[task]["list_dict_result"].append(dict_result)

        # Construct the result for data file, which integrates the raw data, batch data, and result
        for task, dict_task_data in dict_task_data_all.items():
            model_name = dict_task_data["model_name"]
            prompt_mode = dict_task_data["prompt_mode"]
            split = dict_task_data["split"]
            print(f"Task: {task}")
            print(f" - Model Name: {dict_task_data['model_name']}")
            print(f" - Prompt Mode: {dict_task_data['prompt_mode']}")
            print(f" - Split: {dict_task_data['split']}")
            print("----------------------------------------")

            # Merge the raw data, batch data, and result
            list_dict_data = integrate_task_data_batch_result(
                task=task,
                model_name=model_name,
                prompt_mode=prompt_mode,
                split=split,
                dict_task_data=dict_task_data,
                path_dir_raw=path_dir_raw,
                path_dir_batch=path_dir_batch,
            )

            # Save the result
            model_name = model_name.replace("-batch", "")
            path_file_save = os.path.join(
                path_dir_task_save,
                f"{task}/{model_name}/{task}-{prompt_mode}-{decoding_strategy}-{seed}.result.json",
            )
            os.makedirs(os.path.dirname(path_file_save), exist_ok=True)
            with open(path_file_save, "w", encoding="utf-8") as f:
                json.dump(list_dict_data, f, ensure_ascii=False, indent=4)
            print(f" - Saved: {path_file_save}")
            print("========================================")


def integrate_task_data_batch_result(
    task,
    model_name,
    prompt_mode,
    split,
    dict_task_data,
    path_dir_raw="dataset_raw",
    path_dir_batch="azure/input",
):
    """
    Integrarte the task data with the batch result

    Input:
        task: string, the task name
        model_name: string, the model name
        prompt_mode: string, the prompt mode
        split: string, the split of the data
        dict_task_data: dictionary, including model_name, prompt_mode, split, and list_dict_result

    Output:
        list_dict_data: list of dictionary, including id, input, pred, token_input, and token_output

    """
    # Read the raw data file
    path_file_raw = f"{path_dir_raw}/{task}.SFT.json"
    with open(path_file_raw, "r") as f:
        list_dict_data = json.load(f)
    # Filter the data by split
    list_dict_data = [
        dict_data for dict_data in list_dict_data if dict_data["split"] == split
    ]
    print(f" - Num of data: {len(list_dict_data)}")

    # Read the batch data file
    path_file_batch = f"{path_dir_batch}/{model_name}/{prompt_mode}/{task}.batch.jsonl"
    with open(path_file_batch, "r") as f:
        list_dict_batch = [json.loads(line) for line in f]
    print(f" - Num of batch: {len(list_dict_batch)}")

    assert len(list_dict_data) == len(list_dict_batch)

    # Assign the input and output to the data
    for dict_data, dict_batch in zip(list_dict_data, list_dict_batch):
        dict_data["input"] = dict_batch["body"]["messages"]

    # Assign the output to the data
    dict_id_result = {
        dict_result["id"].split("|")[-1]: dict_result
        for dict_result in dict_task_data["list_dict_result"]
    }
    print(f" - Num of result: {len(dict_id_result)}")
    count_match = 0
    for dict_data in list_dict_data:
        id_data = str(dict_data["id"])
        if id_data in dict_id_result:
            dict_result = dict_id_result[id_data]
            dict_data["pred"] = dict_result["output"]
            dict_data["error"] = dict_result["error"]
            count_match += 1
        else:
            print(f" - Missing: {id_data}")
            dict_data["pred"] = ""
            dict_data["error"] = ""

    print(f" - Matched: {count_match}")

    if count_match == len(list_dict_data):
        print(" - All matched.")
    else:
        print(f" - Lost {len(list_dict_data) - count_match} samples.")

    return list_dict_data


def save_azure_batch_file(list_dict_data_batch, path_file_batch):
    """
    Save the list of dictionary to the openai batch file

    Input:
        list_dict_data_batch: list of dictionary, including input, instruction, and model name
        path_file_batch: string, the path of the openai batch file

    Output:
        None

    """
    if "jsonl" not in path_file_batch:
        print("Please use the jsonl format")
        return None
    with open(path_file_batch, "w", encoding="utf-8") as f:
        for dict_data_batch in list_dict_data_batch:
            f.write(json.dumps(dict_data_batch, ensure_ascii=False) + "\n")


def get_azure_price(model_name, flag_batch=False):
    """
    Get the price of the openai model

    Input:
        model_name: string, the model name
        flag_batch: boolean, the flag for batch

    Output:
        cost_input_1m: float, the cost of input per 1M tokens
        cost_output_1m: float, the cost of output per 1M tokens

    """
    dict_model_price = {
        "gpt-4o": [2.5, 10],
        "gpt-4o-mini": [0.15, 0.6],
        "o1-preview": [15, 60],
        "o1-mini": [3, 12],
        "gpt-35-turbo": [0.5, 1.5],
        "gpt-35-turbo-16k": [3, 4],
    }

    cost_input_1m, cost_output_1m = dict_model_price[model_name]
    if flag_batch:
        return cost_input_1m / 2, cost_output_1m / 2
    else:
        return cost_input_1m, cost_output_1m


def cost_calculation_token(
    num_token_input_all,
    num_token_output_all,
    model_name,
    flag_batch=False,
):
    """
    Calculate the cost of the openai model

    Input:
        num_token_input_all: int, the number of input tokens
        num_token_output_all: int, the number of output tokens
        model_name: string, the model name
        flag_batch: boolean, the flag for batch

    Output:
        cost_total: float, the total cost

    """

    cost_input_1m, cost_output_1m = get_azure_price(model_name, flag_batch)
    cost_input = num_token_input_all * cost_input_1m / 10**6
    cost_output = num_token_output_all * cost_output_1m / 10**6
    cost_total = cost_input + cost_output

    return cost_total


def cost_calculation_word(
    list_input,
    list_output,
    flag_batch=False,
    model_name="gpt-4",
):
    """
    Calculate the cost of the openai model

    Input:
        list_input: list of string, the input
        list_output: list of string, the output
        flag_batch: boolean, the flag for batch
        model_name: string, the model name

    Output:
        cost_total: float, the total cost

    """
    tokenizer = tiktoken.encoding_for_model(model_name)
    token_input_each = tokenizer.encode_batch(list_input)
    token_output_each = tokenizer.encode_batch(list_output)
    num_token_input_all = sum(token_input_each)
    num_token_output_all = sum(token_output_each)

    return cost_calculation_token(
        num_token_input_all,
        num_token_output_all,
        model_name,
        flag_batch,
    )


def cost_estimation_word(
    list_input,
    word_output_each,
    num_sample,
    model_name,
    flag_batch=False,
    ratio_word_token=2,
):
    """
    Estimate the cost of the openai model

    Input:
        list_input: list of string, the input
        word_output_each: int, the number of word output each
        num_sample: int, the number of sample
        model_name: string, the model name
        flag_batch: boolean, the flag for batch
        ratio_word_token: int, the ratio of word to token

    Output:
        cost_total: float, the total cost

    """
    # input
    tokenizer = tiktoken.encoding_for_model(model_name)
    token_input_each = tokenizer.encode_batch(list_input)
    num_token_input_all = sum(token_input_each)
    # output
    num_token_output_all = word_output_each * ratio_word_token * num_sample

    return cost_calculation_token(
        num_token_input_all,
        num_token_output_all,
        model_name,
        flag_batch,
    )

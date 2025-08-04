import os
import re
import json
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import glob

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help="Input file path or folder path containing JSON files")
    parser.add_argument('--output_dir', '-o', type=str)
    parser.add_argument('--model_name', '-m', type=str, default="Qwen/Qwen2.5-14B-Instruct",
                        help="HuggingFace model ID for local generation")
    parser.add_argument('--device', '-d', type=int, default=0, help="CUDA device ID (e.g. 0 for V100)")
    return parser.parse_args()

def read_json_file(file_path):
    """Reads a JSON file and returns the data."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {e}")

def save_dict_to_json(dictionary, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(dictionary, json_file, ensure_ascii=False, indent=4)
        print(f"Saved: {file_path}")
    except Exception as e:
        print(f"Failed: {e}")

def get_input_files(input_path):
    """Get list of JSON files to process"""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        # Filter out judgment files to avoid processing them
        json_files = [f for f in json_files if not f.endswith("_judgements.json")]
        return sorted(json_files)
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

def check_and_create_folder(folder_path):
    """
    Check if a folder exists, and create it if it doesn't.

    Args:
        folder_path (str): The path to the folder to check or create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def init_llm(model_name: str, device: int):
    """Initialize the HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def query_llm(model, tokenizer, user_prompt, temperature=0.0, top_p=1.0, max_tokens=1024):
    system_prompt = '''You are a good judge. You will be given a question with list of possible options, a ground truth answer and a model generated response. 
                       You have to determine whether the model generated answer is correct.'''
    
    # Format as chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def extract_judgement(text):
    pattern = r"Explanation: (.*?)\nJudgement: (.*?)(?:\n\n|$)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        explanation = match.group(1)
        judgement = match.group(2)
    else:
        explanation = "No extracted explanation"
        judgement = "No extracted judgement"
    
    results = {"Explanation": explanation, "Judgement": judgement}
    return results


if __name__ == "__main__":
    args = get_args_parser()
    check_and_create_folder(args.output_dir)
    
    # Get list of input files to process
    input_files = get_input_files(args.input)
    print(f"Found {len(input_files)} JSON files to process")
    
    # Initialize model and tokenizer once for all files
    model, tokenizer = init_llm(args.model_name, args.device)

    # the parameter setting used in the original paper
    model_name = args.model_name
    temperature = 0.0
    top_p = 0.9
    max_tokens = 1024

    user_prompt_template = '''
    You will be given a question with list of possible options, a label and a model generated response. Determine whether the model generated response is correct based on the following criteria:
    1. Since there is one and only one correct answer, it should be judged incorrect if the model does not choose any option from the option list or it chooses more than one option.
    2. If the model chooses one option from the option list:
       a. If the ground truth label exactly matches one of the options, the model should be judged correct only if it chooses that exact option.
       b. If the ground truth label is NOT one of the provided options, evaluate whether the model's choice correctly answers the question as posed. For example, if the question asks "which is most closely related" and the model picks the most closely related option from the list, it should be judged correct even if the ground truth label differs.
    3. Notice that there may be some questions that directly ask for the label.
    4. Read the question, options, label and model generated response carefully before making a decision, paying special attention to what the question is actually asking.

    Considering the following examples:
    Question: What is the capital of France? (a) Paris (b) London (c) Berlin (d) Madrid
    Ground truth answer: (a) Paris
    If the model generated response is: "The capital of France is Tokyo.", it should be judged incorrect since it does not choose any option from the option list.
    If the model generated response is: "The capital of France is Paris and London.", it should be judged incorrect since it chooses more than one option from the option list.
    If the model generated response is: "The capital of France is London.", it should be judged incorrect since it chooses one option from the option list but the chosen option does not align with the ground truth answer.
    If the model generated response is: "The capital of France is Paris.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
    Another Question: What is the underlying emotion of the speaker? (a) Happy (b) Sad (c) Angry (d) Neutral
    Ground truth answer: (a) Happy
    If the model generated response is: "The speaker is happy.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
    If the model generated response is: "The speaker expresses happiness.", it should be judged correct since "happiness" aligns with the ground truth answer "happy", and they are just different part of speech of the same word.
    If the model generated response is: "Happiness," it should be judged correct since it is also a valid derivative of the ground truth answer "happy".
    
    Example with ground truth not in options:
    Question: Based on biological classification systems, which of the following animals is most closely related to the one in the audio file? (a) Arctic fox (b) golden jackal (c) black-backed jackal (d) red fox
    Ground truth answer: dog
    If the model generated response is: "The animal in the audio file is most closely related to the black-backed jackal (c).", it should be judged correct since the question asks for the most closely related animal from the given options, and black-backed jackal is indeed the closest relative to a dog among the provided choices, even though the ground truth label is "dog".
    
    Now here is the question and the model generated response for you to judge:
    Question: [QUESTION]
    Ground truth answer: [GROUND_TRUTH_ANSWER]
    Model generated response: [MODEL_GENERATED_RESPONSE]

    Carefully make your decision based on the above criteria. Return your judgement with the following format:
    Explanation: <Your explanation on your judgement>
    Judgement: <Your judgement, either "correct" or "incorrect">
    '''

    # Process each input file
    for input_file in input_files:
        # Check if judgements file already exists
        output_filename = os.path.basename(input_file).replace(".json", "_judgements.json")
        output_path = os.path.join(args.output_dir, output_filename)
        
        if os.path.exists(output_path):
            print(f"\nSkipping {input_file} - judgements already exist at {output_path}")
            continue
            
        print(f"\nProcessing: {input_file}")
        model_responses = read_json_file(input_file)
        judgements = {}
        judgements['judge_model'] = model_name
        judgements['temperature'] = temperature
        judgements['top_p'] = top_p
        judgements['max_tokens'] = max_tokens
        judgements['results'] = {}
        
        for wav_file in tqdm(model_responses['results'].keys(), desc=f"Judging {os.path.basename(input_file)}"):
            question = model_responses['results'][wav_file]['instruction']
            response = model_responses['results'][wav_file]['response']
            ground_truth_answer = model_responses['results'][wav_file]['label']
            user_prompt = user_prompt_template.replace("[QUESTION]", question).replace("[MODEL_GENERATED_RESPONSE]", response).replace("[GROUND_TRUTH_ANSWER]", ground_truth_answer)

            judgement = query_llm(model, tokenizer, user_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            results = extract_judgement(judgement)

            results['instruction'] = question
            results['model_response'] = response
            results['label'] = ground_truth_answer
            judgements['results'][wav_file] = results
        
        # Save judgements for this file
        save_dict_to_json(judgements, output_path)
    
    print(f"\nCompleted processing {len(input_files)} files")
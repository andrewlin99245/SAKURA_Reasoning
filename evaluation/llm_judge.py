import os
import re
import json
from tqdm import tqdm
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help="Input file path")
    parser.add_argument('--output_dir', '-o', type=str, help="Output directory")
    parser.add_argument('--model_name', '-m', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="HuggingFace model ID for local generation")
    parser.add_argument('--device', '-d', type=int, default=0, help="CUDA device ID (e.g. 0 for V100)")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    return parser.parse_args()


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_dict_to_json(dictionary, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
    print(f"Saved: {file_path}")


def check_and_create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    print(f"Folder '{folder_path}' is ready.")


def init_llm(model_name: str, device: int, temperature: float, top_p: float, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens
    )
    return gen


def query_llm(generator, prompt: str) -> str:
    out = generator(prompt, return_full_text=False, do_sample=False)
    return out[0]['generated_text']


def extract_judgement(text: str) -> dict:
    pattern = r"Explanation: (.*?)\nJudgement: (.*?)(?:\n\n|$)"
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return {"Explanation": m.group(1).strip(), "Judgement": m.group(2).strip()}
    return {"Explanation": "No extracted explanation", "Judgement": "No extracted judgement"}


if __name__ == '__main__':
    args = get_args_parser()
    check_and_create_folder(args.output_dir)

    # Initialize local LLM pipeline
    llm = init_llm(
        model_name=args.model_name,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens
    )

    data = read_json_file(args.input)
    judgements = {
        'judge_model': args.model_name,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_tokens': args.max_new_tokens,
        'results': {}
    }

    base_template = '''
You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
1. Since there is one and only one correct answer, it should be judged incorrect if the model does not choose any option from the option list or it chooses more than one option.
2. If the model chooses one option from the option list, it should be judged correct if the chosen option aligns with the ground truth answer, otherwise it should be judged incorrect.
3. Read the question, options, ground truth answer and model generated response carefully before making a decision.

Now here is the question and the model generated response for you to judge:
Question: {question}
Ground truth answer: {ground_truth}
Model generated response: {response}

Return your answer in the format:
Explanation: <...>\nJudgement: <correct|incorrect>'''

    for wav_file, entry in tqdm(data['results'].items()):
        question = entry['instruction']
        response = entry['response']
        ground_truth = entry['label']
        prompt = base_template.format(
            question=question,
            ground_truth=ground_truth,
            response=response
        )
        raw = query_llm(llm, prompt)
        res = extract_judgement(raw)
        res.update({
            'instruction': question,
            'model_response': response,
            'label': ground_truth
        })
        judgements['results'][wav_file] = res

    out_path = os.path.join(args.output_dir, os.path.basename(args.input).replace('.json', '_judgements.json'))
    save_dict_to_json(judgements, out_path)
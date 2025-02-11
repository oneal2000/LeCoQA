import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "/liuzyai04/thuir/yuebaoqing/LLM/Baichuan-13B-Chat",
    use_fast=False,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/liuzyai04/thuir/yuebaoqing/LLM/Baichuan-13B-Chat",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.generation_config = GenerationConfig.from_pretrained(
    "/liuzyai04/thuir/yuebaoqing/LLM/Baichuan-13B-Chat"
)

# Function to generate a response to a given question
def generate_reasoning(question):
    messages = [{"role": "user", "content": question}]
    response = model.chat(tokenizer, messages)
    return response

# Function to process a dataset
def process_dataset(dataset_path, output_path):
    # Load the dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Generate reasoning and save results
    test_result = []
    for one in tqdm(data, desc="Processing"):
        gen_ans = generate_reasoning(one['问题'])
        exp_ans = one['答案文本']
        print(f"Generated answer: {gen_ans}")

        # Update data and add to results list
        entry = {
            "gen_ans": gen_ans,
            "exp_ans": exp_ans
        }
        test_result.append(entry)

    # Save the results
    with open(output_path, 'w') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")

# Example usage
dataset_path = "../data/test.json"  # Specify your dataset path here
output_path = "./output/zeroshot_bc.json"  # Specify your output path here
process_dataset(dataset_path, output_path)

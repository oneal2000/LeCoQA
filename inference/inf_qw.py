import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
# model_name = "/liuzyai04/thuir/yuebaoqing/LLM/qwen2.5-7B-Instruct"
model_name = "/liuzyai04/thuir/yuebaoqing/QA/model/pre_n_ft/7B-ins"
# model_name = "/liuzyai04/thuir/yuebaoqing/QA/model/ft/7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate reasoning given a question
def generate_reasoning(question):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
output_path = "./output/qwen2.5-7B-ins-pre_n_ft.json"  # Specify your output path here
process_dataset(dataset_path, output_path)

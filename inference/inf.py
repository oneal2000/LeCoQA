import json
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import copy

fewshot_prefix = """
你是一个具备法律专业知识的法律助手，请先仔细阅读下面的三个例子：
### 示例 1:
**问题**: 谁可以成为个体工商户？  
**模型回答**:  
您好，根据有关法律的规定，有经营能力的公民经过登记可以成为个体工商户。  
我国个体工商户对港澳台地区居民有限制地开放。外国公民目前不可以申请在中国成立个体工商户进行商事经营。

---

### 示例 2:
**问题**: 合伙必须要签订书面合同吗？  
**模型回答**:  
您好，1.《中华人民共和国民法典》并未规定合伙合同必须要采取书面形式，因此未采取书面形式不影响合伙合同的成立和生效。但需要注意的是，如果未采取书面形式，仅以口头形式达成合伙合同，可能在诉讼中难以举证证明合同关系存在，进而导致不利的诉讼后果。因此，虽然法律未强制规定合伙合同一定要采取书面形式，但稳妥起见还是建议签订书面合伙协议。\n2.但是需要注意的是，如果是想设立合伙企业，那么根据《合伙企业法》第四条的规定，合伙协议由全体合伙人协商一致，必须以书面形式签订合伙协议。

---

### 示例 3:
**问题**: 未发生法律效力的征收补偿协议，是否属于行政诉讼受案范围？     
**模型回答**:  
您好，未发生法律效力的征收补偿协议，对当事人的权利义务不产生实际影响，不属于行政诉讼的受案范围。在集体土地预征收过程中，征收管理部门与被征收人签订的征收补偿协议属于附条件的行政行为，只有在省级人民政府作出征收批复，市县人民政府发布正式的征收公告后，征收补偿协议才能够发生法律效力，对当事人的权利义务产生实际影响。未发生法律效力的征收补偿协议，对当事人的权利义务不产生实际影响，不属于行政诉讼的受案范围。
---
### 任务:
模仿上面的三个例子的形式，简明扼要地回答如下问题。

**问题**: """

# Define different prompt templates
PROMPT_TEMPLATES = {
    "direct": [
        {"role": "system", "content": "你是一个法律助理，提供帮助。"},
        {"role": "user", "content": "{question}"}
    ],
    "fewshot": [
        {"role": "system", "content": "你是一个法律助理，提供帮助。"},
        {"role": "user", "content": fewshot_prefix + "{question}"}
    ]
}

def generate_reasoning(question, tokenizer, model, baseline):
    messages = copy.deepcopy(PROMPT_TEMPLATES[baseline]) # 要用深拷贝而不能直接用，否则会污染模版，导致question无法更新
    messages[-1]["content"] = messages[-1]["content"].format(question=question)
    
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

def process_dataset(dataset_path, output_path, model_name, baseline):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    # Generate reasoning and save results
    with open(output_path, 'w') as f:
        for one in tqdm(data, desc="Processing"):
            gen_ans = generate_reasoning(one['问题'], tokenizer, model, baseline)
            entry = {
                "id": str(one['query_id']),
                "answer": gen_ans
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning from a dataset using a specified model and prompt baseline.")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the pre-trained model.")
    parser.add_argument("--baseline", type=str, choices=["direct", "fewshot"], required=True, help="Prompt template to use: 'direct' or 'fewshot'.")
    parser.add_argument("--dataset_path", type=str, default="../data/example/test.json")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results.")
    
    args = parser.parse_args()
    process_dataset(args.dataset_path, args.output_path, args.model_name, args.baseline)

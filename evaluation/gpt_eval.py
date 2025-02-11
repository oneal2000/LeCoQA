from openai import OpenAI
import time
import httpx
import json
from tqdm import tqdm
import os
from multiprocessing import Pool

client = OpenAI(
    base_url="", 
    api_key="",  
    http_client=httpx.Client(
        base_url="",
        follow_redirects=True,
    ),
)


def chat(content,max_retries = 5):
    attempt = 0
    response = None
    while attempt < max_retries:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            response = completion.choices[0].message.content
            break
        except:
            attempt += 1
            time.sleep(2)  
    return response


def process_file(input_file, output_file, query_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    queries = []
    laws = {}
    with open(query_file, 'r') as qfile:
        content = json.load(qfile)
        for i, tmp in enumerate(content):
            queries.append(tmp['问题'])
            laws[i] = list(tmp['相关法规'].values())

    result = []
    with open(output_file, 'w') as outfile:
        for i, item in enumerate(tqdm(data,desc='doing')):
            gen_ans = item['gen_ans']
            exp_ans = item['exp_ans']
            
            prompt = f"""
问题描述：{queries[i]}

相关法律条文：{laws[i]}

标准答案：{exp_ans}

待评测的答案：{gen_ans}

评估任务：请参考相关法律条文，比较待评测的答案与标准答案的一致性。大意大致相同即可视为”对“。

输出格式：
请只回答一个字：对/错，表明待评测答案是否正确。切勿回答其它内容。
"""
            # print(prompt)
            output = chat(prompt)
            # print(output)
            outfile.write(output + '\n')
        
    
    print(f"Results saved to {output_file}")

def process_wrapper(input_file, output_file, query_file):
    """封装处理函数以适配多进程调用"""
    print(f"Start processing {input_file}")
    process_file(input_file, output_file, query_file)
    print(f"Finished processing {input_file}")
    
    
if __name__ == "__main__":
    input_folder = "data/modified"
    output_folder = "LLM-Eval"
    query_file = os.path.join(input_folder, "queries.json")
    
    tasks = []
    for filename in os.listdir(input_folder):
        # 跳过非 JSON 文件或特定文件（如 queries.json）
        if not filename.endswith(".json") or filename == "queries.json" or filename.startswith("zero"):
            continue
        # if not filename == "qwen2.5-7B-ins-pre_n_ft.json":
        #     continue
        
        # 设置文件路径
        input_file = os.path.join(input_folder, filename)
        outfile_name = filename[:-5] + ".txt"
        output_file = os.path.join(output_folder, outfile_name)

        # 添加任务
        tasks.append((input_file, output_file, query_file))

    # 并行处理
    with Pool(processes=4) as pool:
        pool.starmap(process_wrapper, tasks)
    
        
        
    
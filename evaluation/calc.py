import jieba
from rouge_score import rouge_scorer
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
from nltk.translate.meteor_score import meteor_score

# 加载 JSON 数据
with open('your_file.json', 'r') as file:
    data = json.load(file)

# 初始化指标存储
results = {
    "METEOR": [],
    "BERTScore": [],
    "BLEU": [],
    "ROUGE": [],  # 新增综合 ROUGE 指标
}

# 初始化 ROUGE 评分器
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)  # 中文无需词干化

# BLEU 平滑函数
smoothie = SmoothingFunction().method4

# 逐条计算指标
for item in data:
    exp_ans = item["exp_ans"]
    gen_ans = item["gen_ans"]
    
    # 计算METEOR分数
    reference_tokens = list(jieba.cut(exp_ans))
    hypothesis_tokens = list(jieba.cut(gen_ans))
    score_meteor = meteor_score([reference_tokens], hypothesis_tokens)
    results["METEOR"].append(score_meteor)

    # 计算 ROUGE 分数
    rouge_scores = scorer.score(" ".join(jieba.cut(exp_ans)), " ".join(jieba.cut(gen_ans)))
    rouge1 = rouge_scores['rouge1'].fmeasure
    rouge2 = rouge_scores['rouge2'].fmeasure
    rougeL = rouge_scores['rougeL'].fmeasure

    # 计算综合 ROUGE 指标（加权平均）
    rouge_combined = (rouge1 + rouge2 + rougeL) / 3  # 等权重
    results["ROUGE"].append(rouge_combined)

    # 分词处理
    exp_ans_tokens = list(jieba.cut(exp_ans))
    gen_ans_tokens = list(jieba.cut(gen_ans))

    # 计算 BLEU 分数
    bleu_score = sentence_bleu([exp_ans_tokens], gen_ans_tokens, smoothing_function=smoothie)
    results["BLEU"].append(bleu_score)

# 使用本地模型路径计算 BERTScore
local_model_path = "bert-base-chinese"  # Hugging Face 模型名称
gen_ans_list = [" ".join(jieba.cut(item["gen_ans"])) for item in data]
exp_ans_list = [" ".join(jieba.cut(item["exp_ans"])) for item in data]

P, R, F1 = score(gen_ans_list, exp_ans_list, model_type=local_model_path, verbose=True)
results["BERTScore"] = F1.tolist()  # 转为普通列表方便查看

# 计算并输出每个指标的平均值
print("平均指标：")
for metric, scores in results.items():
    mean_score = sum(scores) / len(scores)
    print(f"{metric}: Mean = {mean_score:.4f}")

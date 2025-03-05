import jieba
import json
import argparse
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

class Evaluator:
    def __init__(self, gen_file, exp_file):
        self.gen_data = self.load_data(gen_file)
        self.exp_data = self.load_data(exp_file)
        assert self.gen_data.keys() == self.exp_data.keys(), "Mismatch between gen_data and exp_data keys"
        
        self.results = {
            "METEOR": [],
            "BERTScore": [],
            "BLEU": []
        }
        self.smoothie = SmoothingFunction().method4
    
    def load_data(self, file_path):
        """加载 JSONL 文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return {item['id']: item['answer'] for item in (json.loads(line) for line in file)}
    
    def calculate_metrics(self, exp_text, gen_text):
        """计算各项评估指标"""
        # METEOR
        reference_tokens = list(jieba.cut(exp_text))
        hypothesis_tokens = list(jieba.cut(gen_text))
        self.results["METEOR"].append(meteor_score([reference_tokens], hypothesis_tokens))
        
        # BLEU
        self.results["BLEU"].append(sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=self.smoothie))
    
    def calculate_bert_score(self):
        """计算 BERTScore"""
        local_model_path = "bert-base-chinese"
        gen_ans_list = [" ".join(jieba.cut(self.gen_data[idx])) for idx in self.gen_data]
        exp_ans_list = [" ".join(jieba.cut(self.exp_data[idx])) for idx in self.exp_data]
        
        _, _, F1 = score(gen_ans_list, exp_ans_list, model_type=local_model_path)
        self.results["BERTScore"] = F1.tolist()
    
    def run(self):
        """执行评测流程"""
        for idx in self.exp_data:
            self.calculate_metrics(self.exp_data[idx], self.gen_data[idx])
        self.calculate_bert_score()
        self.print_results()
    
    def print_results(self):
        """打印计算出的指标均值"""
        print("平均指标：")
        for metric, scores in self.results.items():
            mean_score = sum(scores) / len(scores) if scores else 0
            print(f"{metric}: Mean = {mean_score:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="计算 NLP 评测指标")
    parser.add_argument('--gen_file', type=str, required=True, help='生成数据 JSONL 文件路径')
    parser.add_argument('--exp_file', type=str, required=True, help='真实数据 JSONL 文件路径')
    args = parser.parse_args()
    
    evaluator = Evaluator(args.gen_file, args.exp_file)
    evaluator.run()

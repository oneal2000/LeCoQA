# LeCoQA: A Chinese Legal Consultation QA Dataset



## Overview

##TODO##



## Installation Instructions

### Requirements

Before you begin, make sure you have the following packages installed in your environment:

```plaintext
jieba==0.42.1
numpy==1.26.4
pandas==2.2.2
torch==2.0.1
tqdm==4.66.5
transformers==4.39.3
deepspeed==0.10.0
nltk==3.9.1
rouge-score==0.1.2
bert_score==0.3.9
```



### Setting Up Your Environment

To create a new environment and install the required packages, follow these steps:

```
conda create -n lecoqa python=3.9
conda activate lecoqa
pip install -r requirements.txt
```



**Note:** The `requirements.txt` file should exclude`torch`. Install PyTorch specifically according to your system setup by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

```
pip install torch==2.0.1
```



## Dataset Structure

### Directory Overview

The root directory for the LeCoQA dataset is located at `/LeCoQA/data`. The dataset comprises a total of 1,543 query-answer pairs and a corresponding large-scale corpus of 55,348 statutory articles.

### Data Files and Structure

**Queries:** The queries and their relevant answers are stored in the JSON file:

```
data/queries.json
```



**Example Query:** Below is a sample entry from `queries.json`, showcasing the structure and data fields:

```
{
    "问题": "谁可以成为个体工商户？",
    "关键词": "个体工商户资格",
    "答案文本": "您好，根据有关法律的规定，有经营能力的公民经过登记可以成为个体工商户。\n我国个体工商户对港澳台地区居民有限制地开放。外国公民目前不可以申请在中国成立个体工商户进行商事经营。",
    "相关法规": {
        "中华人民共和国民法典第五十四条": "自然人从事工商业经营，经依法登记，为个体工商户。个体工商户可以起字号。\n",
        "个体工商户条例第二条": "有经营能力的公民，依照本条例规定经工商行政管理部门登记，从事工商业经营的，为个体工商户。\n\n个体工商户可以个人经营，也可以家庭经营。\n\n个体工商户的合法权益受法律保护，任何单位和个人不得侵害。\n"
    },
    "match_id": [
        705,
        49282
    ],
    "match_name": [
        "中华人民共和国民法典第五十四条",
        "个体工商户条例第二条"
    ],
    "query_id": 0
},

Translated:
{
    "Question": "Who can become an individual industrial and commercial household?",
    "Keywords": "individual industrial and commercial household eligibility",
    "Answer": "Hello, according to relevant laws, citizens with the capacity to operate a business can become individual industrial and commercial households after registration.\nIn China, the eligibility for individual industrial and commercial households is partially open to residents from Hong Kong, Macau, and Taiwan. Foreign citizens are currently not allowed to apply to establish an individual industrial and commercial household in China for commercial operations.",
    "Evidence": {
        "Article 54 of the Civil Code of the People's Republic of China": "A natural person engaging in industrial and commercial activities, registered according to the law, is an individual industrial and commercial household. An individual industrial and commercial household may adopt a trade name.\n",
        "Article 2 of the Individual Industrial and Commercial Household Regulation": "Citizens with the capacity to operate a business, upon registration with the administrative department for industry and commerce as per the provisions of this regulation, engaging in industrial and commercial activities, are individual industrial and commercial households.\n\nIndividual industrial and commercial households can be operated by an individual or by a family.\n\nThe lawful rights and interests of individual industrial and commercial households are protected by law, and no organization or individual may infringe upon them.\n"
    },
    "match_id": [
        705,
        49282
    ],
    "match_name": [
        "Article 54 of the Civil Code of the People's Republic of China",
        "Article 2 of the Individual Industrial and Commercial Household Regulation"
    ],
    "query_id": 0
}
```



**Corpus:** The comprehensive corpus containing all statutory articles is available in:

```
data/corpus.json
```



**Example Article:** Here is an example from `corpus.jsonl`, illustrating the format and content:

```
{
	"id": 22648,
	"name": "农村土地承包经营纠纷调解仲裁法第十八条",
	"content": "农村土地承包经营纠纷申请仲裁的时效期间为二年，自当事人知道或者应当知道其权利被侵害之日起计算。\\n"
}

Translated:
{
    "id": 22648,
    "name": "Article 18 of the Law on Mediation and Arbitration of Disputes over Rural Land Contracting Management",
    "content": "The limitation period for applying for arbitration of disputes over rural land contracting management is two years, calculated from the date when the party becomes aware or should have become aware of the infringement of their rights."
}
```



### Data Collection Methodology

This collection includes national-level Chinese laws, regulations, and judicial interpretations. Our legal team has meticulously gathered the latest versions from official government platforms. The documents are organized into the smallest searchable units by article, allowing for detailed legal research and practical application.

## Evaluation

Our evaluation framework tests the dataset using multiple QA-pair generating methods, including Zero-shot, Few-shot, RAG, Finetune, Pretrain + Finetune and Few-shot + Evidence. We standardize our evaluation process by pre-processing the original queries and partitioning the dataset into two subsets: `train` and `test`. All related files are located in `data/example`.

### Dataset Files

Here are the files associated with the training and testing datasets, the datasets `train` and `test` are randomly split in a 4:1 ratio. 

```
data/example/train.json # used for training
data/example/test.json # used for testing
```

| **STATISTICS**                          | **NUMBER**   |
| --------------------------------------- | ------------ |
| Train Questions                         | 1234         |
| Test Questions                          | 309          |
| Average Len of Questions                | 27.31        |
| Average Len of Evidence                 | 126.80       |
| Average Len of Answers                  | 231.32       |
| Size of Corpus                          | 55348        |
| Average Number of Evidence per Question | 1.7608554763 |



### MODELS

The experiments were carried out on four white-box models and one black-box model. The white-box models include **Baichuan-13B-Chat**, **qwen2.5-7B**, **qwen2.5-7B-Instruct**, and **qwen2.5-14B-Instruct**. Due to computational limitations, we only trained the models requiring training on the two 7B models. Our black-box model is the **GPT-3.5-turbo**.



### METRICS

We conducted evaluation experiments on this dataset to assess the performance of various methods in generating answers to questions and measuring their similarity to correct answers. Specifically, we used the following four metrics:

- **LLM-Eval**: Utilizes the GPT-3.5-turbo API to evaluate whether the answers provided are correct. This approach leverages the capabilities of a large-scale language model to assess the quality and correctness of generated responses.

- **METHOR**: A metric designed to measure the effectiveness of generated answers based on their thematic and contextual relevance. It evaluates how well the response aligns with the expected themes and context of the question.

- **BertScore**: A semantic evaluation metric that uses BERT embeddings to calculate similarity scores between generated answers and reference answers. BertScore considers the contextual word embeddings, providing a more nuanced assessment of similarity than traditional lexical-based metrics.

- **BLEU**: A widely used metric for evaluating the quality of text which measures the n-gram overlap between generated outputs and reference texts. Primarily used in machine translation evaluation, BLEU considers precision in n-gram matches, thus evaluating how closely the generated text matches the reference answer.



### METHODS

The methods of generation are as follows:

- **Zero-shot**: A method where the model is expected to generate answers without any prior specific training examples. The model relies on its pre-existing knowledge to respond to the questions.

- **Few-shot**: The model is provided with a small set of examples or prompts (3 as default in our experiment) to generate answers, helping it better understand the task. 

- **RAG (Retrieval-Augmented Generation)**: This approach combines retrieval mechanisms with generation capabilities. The model first retrieves relevant legal evidence related to the query and then uses this information to craft more informed answers. In our experiments, we recall ten examples per query.

- **Finetune**: Involves finetuning the models on our dataset to optimize its performance in answering legal-specific questions, enhancing its ability to generate contextually relevant answers.

- **Pretrain + Finetune**: The model is initially pretrained on a large corpus of legal articles  to imbue it with extensive legal knowledge, followed by fine-tuning on the specific dataset to hone its response capabilities for the given queries.

- **Few-shot + Evidence**: An extension of the few-shot approach, this method supplements the examples with additional evidence to solidify the grounding of the generated answers.

The experimental results illustrate the capacity of these models under various configurations, demonstrating the practical applicability of machine learning techniques in the field of legal research.



### Experiment Results 



| Model                | Setup              | LLM-Eval | METEOR | BertScore | BLEU  |
|----------------------|--------------------|---------:|-------:|----------:|------:|
| Baichuan-13B-chat    | Direct             |   0.0939 | 0.2293 |    0.6836 | 0.0442 |
|                      | Few-shot           |   0.1165 | 0.2212 |    0.6964 | 0.0537 |
|                      | RAG                |   0.1003 | 0.2323 |    0.7028 | 0.0743 |
|                      | Few-shot + Evidence| **0.1327** | **0.2421** | **0.7048** | **0.0786** |
| qwen2.5-7B-Instruct  | Direct             |   0.1392 | 0.2320 |    0.6887 | 0.0521 |
|                      | Few-shot           |   0.1618 | 0.1735 |    0.6901 | 0.0340 |
|                      | RAG                |   0.1683 | 0.2335 |    0.7278 | 0.0897 |
|                      | Finetune           |   0.1586 | 0.2488 |    0.7279 | 0.0945 |
|                      | Pretrain + Finetune|   0.1715 | **0.2518** |    0.7289 | **0.0963** |
|                      | Few-shot + Evidence|   <u>**0.2718**</u> | 0.2442 |    **0.7351** | 0.0916 |
| qwen2.5-14B-Instruct | Direct             |   0.0841 | **0.2293** |    0.6696 | **0.0373** |
|                      | Few-shot           |   0.1748 | 0.1811 |    0.6900 | **0.0373** |
|                      | RAG                |   **0.1877** | 0.1788 |    **0.6923** | 0.0364 |
|                      | Few-shot + Evidence|   0.1489 | 0.1789 |    0.6913 | 0.0341 |
| qwen2.5-7B           | Finetune           |   **0.1941** | **0.2671** |    0.7334 | **0.1056** |
|                      | Pretrain + Finetune|   0.1877 | 0.2612 |    **0.7335** | 0.1054 |
|                      | Few-shot + Evidence|   0.0615 | 0.2044 |    0.6801 | 0.0554 |
| Black Box LLM (GPT)  | Direct   |   **0.2686** | 0.2382 |    0.7003 | 0.0474 |
|                      | Few-shot           |   0.1942 | 0.2288 |    0.7147 | 0.0533 |
|                      | RAG                |   0.1845 | 0.2974 |    0.7437 | 0.1012 |
|                      | Few-shot + Evidence|   0.2460 | <u>**0.3136**</u> |    <u>**0.7524**</u> | <u>**0.1152**</u> |

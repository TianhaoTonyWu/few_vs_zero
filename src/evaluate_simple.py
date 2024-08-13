import argparse
import json
import torch
from vllm import LLM, SamplingParams
import os
from utils import process_and_save_tokens, evaluate_results
from transformers import AutoTokenizer

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/nfs20t/songran/llm/llama-7b-hf-chat")
parser.add_argument("-ts", "--shot", type=int, default=0)
parser.add_argument("-t", "--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-d", "--device", type=str, default="6")
parser.add_argument("-md", "--mod", type=str, default="GV_trace-test")
args = parser.parse_args()

# Set system variables
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 获取任务和model名称
task = args.task.split("_")[0]
basemodel = args.model.split('/')[-1]

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

train_token, test_token, labels = process_and_save_tokens(
    task=args.task, # .json file name in the dataset
    shot=args.shot,
    tokenizer=tokenizer
)

# 加载模型
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)

# 生成结果
outputs = model.generate(prompt_token_ids=test_token, sampling_params=SamplingParams(max_tokens=20,temperature=0,top_p=1,stop="[INST]"))

output_folder = f"results/{basemodel}/{task}/"
os.makedirs(output_folder, exist_ok=True)
output_file = f"{output_folder}/{task}-{args.shot}-shot.jsonl"

# evalaute and save results
num, correct, accuracy, avg_rougeL = evaluate_results(output_file=output_file, tokenizer=tokenizer, labels=labels, outputs=outputs)
   
# Record summary
with open(f"new_result_0813.jsonl","a") as f:
    f.write("----------------------------\n")
    f.write(f"option:{task},mod:{args.mod},task:{task},shot:{args.shot}\n")
    f.write(json.dumps({"num": num, "correct": correct, "acc": accuracy, "avg_rougeL": avg_rougeL}, ensure_ascii=False) + "\n")
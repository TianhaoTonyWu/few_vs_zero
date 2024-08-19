import json
import random
import os
import string
import pickle
from tqdm import tqdm
from jinja2 import Template
from rouge_score import rouge_scorer

def find_all_sublists(main,sub):
    # 获取子列表的长度
    sub_len = len(sub)
    # 存储所有匹配子列表的起始索引
    indices = []
    # 遍历主列表，长度减去子列表的长度
    for i in range(len(main) - sub_len + 1):
        # 如果在主列表中找到与子列表匹配的部分
        if main[i:i+sub_len] == sub:
            indices.append([j+1 for j in range(i,i+sub_len)])  # 添加索引到列表
    return indices  # 返回所有匹配的索引列表

def random_shot(data,shot):
     random_selection = random.sample(data, shot)
     return random_selection

def data_construct(data, instruction, shot=3):
    
    out_data = []
    instruction = "".join(instruction)

    for i in data:
        messages = [{"role":"system","content":instruction}]
        shot_content = random_shot(data, shot=shot)
        for j in shot_content:
            messages.append({"role":"user","content":"".join(j["input"])})
            messages.append({"role":"assistant","content":"".join(j["output"])})
        messages.append({"role":"user","content":"".join(i["input"])})
        messages.append({"role":"assistant","content":"".join(i["output"])})
        out_data.append(messages)
    return out_data

def clean_text(text):
    return text.strip().lower().rstrip(string.punctuation)


def process_and_save_tokens(task, shot, tokenizer):
    """
    Process and save train and test tokens along with labels.

    Parameters:
        task (str): The task name to be used for file naming.
        task_shot (int): The number of shots used for the task.
        train_message (list): The training messages.
        test_message (list): The testing messages.
        tokenizer (AutoTokenizer): The tokenizer used to encode messages.
        base_path (str): The base directory for saving token files.
    
    Returns:
        train_token (list): List of processed train tokens.
        test_token (list): List of processed test tokens.
        labels (list): List of labels corresponding to the test tokens.
    """

    # DATA_PATH = "/home/wutianhao/project/few_vs_zero/natural-instructions-master/tasks/"
    DATA_PATH = "/home/wutianhao/project/few_vs_zero/task_original_backup"
    base_path='/home/wutianhao/project/few_vs_zero/data/data_token/'
    task_name = task.split("_")[0]


    # Load data and split into training and test sets
    data_file = os.path.join(DATA_PATH, task)
    with open(data_file, "r") as f:
        data = json.load(f)
        instruction = data["Definition"]
        instances = data["Instances"][:6500]
        data_number = len(instances)
        train, test = instances[:data_number // 2], instances[data_number // 2:]
        train_message = data_construct(train, instruction, shot=shot)
        test_message = data_construct(test, instruction, shot=shot)
    """
    # Process and save train tokens
    train_file = os.path.join(base_path, task_name, f'train_{shot}.pkl')
    if os.path.exists(train_file):
        with open(train_file, 'rb') as f:
            train_token = pickle.load(f)["inputs"]
    else:
        train_token = []
        progress_bar = tqdm(total=len(train_message), desc='Train Processing data')
        for message in train_message:
            progress_bar.update(1)
            template_str = tokenizer.default_chat_template
            template = Template(template_str)
            result = template.render(messages=message, bos_token="", eos_token="")
            train_token.append(tokenizer.encode(result))
        progress_bar.close()

        os.makedirs(os.path.dirname(train_file), exist_ok=True)
        with open(train_file, 'wb') as f:
            pickle.dump({"inputs": train_token}, f)
    """
    # Process and save test tokens and labels
    test_file = os.path.join(base_path, task_name, f'test_{shot}.pkl')
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
            test_token, labels = data["inputs"], data["labels"]
    else:
        test_token, labels = [], []
        progress_bar = tqdm(total=len(test_message), desc='Test Processing data')
        for message in test_message:
            progress_bar.update(1)
            prompt, output = message[:-1], message[-1]
            template_str = tokenizer.default_chat_template
            template = Template(template_str)
            result = template.render(messages=prompt, bos_token="", eos_token="")
            test_token.append(tokenizer.encode(result))
            labels.append(output["content"])
        progress_bar.close()

        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'wb') as f:
            pickle.dump({"inputs": test_token, "labels": labels}, f)

    # Train token is never used in current setting
    return None, test_token, labels



def evaluate_results(output_file, tokenizer, labels, outputs):
    # 创建 rouge
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with open(output_file, "w", encoding="utf8") as f:
        # 使用列表推导式简化对 data_output 列表的构建
        data_output = [{"prompt": tokenizer.decode(output.prompt_token_ids),
                        "pred": output.outputs[0].text,
                        "label": l}
                    for l, output in zip(labels, outputs)]

    
        correct = 0
        total_rougeL = 0.0

        for j in data_output:
            #  清理结果
            clean_pred = clean_text(j["pred"])
            clean_label = clean_text(j["label"])

            # 计算 EM
            if clean_pred == clean_label:
                correct+=1

            # 计算 ROUGE-L
            rougeL_score = scorer.score(clean_label, clean_pred)['rougeL'].fmeasure
            total_rougeL += rougeL_score
            j["rougeL"] = rougeL_score
            
            # 将 data_output 中的数据以 JSON 格式写入文件
            f.write(json.dumps(j,ensure_ascii=False) + "\n")

        # 计算tong结果
        num = len(data_output)
        avg_rougeL = total_rougeL / num
        accuracy = correct / num

        f.write(json.dumps({"num": num,"correct": correct,"acc": accuracy,"avg_rougeL": avg_rougeL},ensure_ascii=False) + "\n")

        return num, correct, accuracy, avg_rougeL
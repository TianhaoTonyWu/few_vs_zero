from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import os
from utils import data_construct,find_all_sublists
from transformers import AutoTokenizer
import pickle
from tqdm import tqdm 
import os
from jinja2 import Template
import pickle
import argparse
import  torch.nn as nn
import json
import chat_template

## 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/root/autodl-tmp/model")
parser.add_argument("-t","--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-d","--device",type=str, default="0,1")
parser.add_argument("-st","--shot",type=int, default=5)
parser.add_argument("-md", "--mod", type=str, default="GV_trace-test")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device


# 子序列 token
sub_squence = {
    "[INST]":[518, 25580, 29962],
    "[/INST]":[518, 29914, 25580, 29962],
    "<<SYS>>":[3532, 14816, 29903, 6778],
    "<</SYS>>":[529, 829, 14816, 29903, 6778]
}
sub_squence_list = [[518, 25580, 29962],[518, 29914, 25580, 29962],[3532, 14816, 29903, 6778],[529, 829, 14816, 29903, 6778]]



DATA_PATH = "/root/project/few_vs_zero/task_original_backup"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 获取任务名称
task = args.task.split("_")[0]

# 加载数据并拆分为训练集和测试集
data_path = os.path.join(DATA_PATH, args.task)
with open(data_path, "r") as f:
    data = json.load(f)
    instruction = data["Definition"]
    instance = data["Instances"]
    data_number = len(instance)
    train, test = instance[:data_number//2], instance[data_number//2:]
    train_message = data_construct(train, instruction, shot=args.shot)
    test_message = data_construct(test, instruction, shot=args.shot)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.chat_template = chat_template.ct
print(tokenizer.get_chat_template())
# 处理训练数据并保存token
if "trace"  not in args.mod:
    train_file = f'/root/project/few_vs_zero/data/data_token/{task}/train_{str(args.shot)}.pkl'
else:
    train_file = f'/root/project/few_vs_zero/data/data_token/{task}/train_trace_{str(args.shot)}.pkl'
if os.path.exists(train_file):
    with open(train_file, 'rb') as f:
        data = pickle.load(f)
        train_token = data["inputs"]
        if "indexs" in list(data.keys()):
            indexs = data["indexs"]
else:
    ## 如果文件不存在则处理新的数据
    train_token = []
    indexs = []
    progress_bar = tqdm(total=len(train_message), desc='Train Processing data')
    for i in range(len(train_message)):
        progress_bar.update(1)
        message = train_message[i]
        template_str = tokenizer.chat_template
        template = Template(template_str)
        bos_token = ""
        eos_token = ""
        result = template.render(messages=message, bos_token=bos_token, eos_token=eos_token).replace("<spe>"," ")

        ## 如果是trace模式，则提取子序列的索引
        if "trace" in args.mod:
            input_ids = tokenizer.encode(result)
            if "trace-test" in args.mod:
                # Every [INST] except 1st
                index_start = find_all_sublists(input_ids,sub_squence_list[0])[1:]
                # Every [/INST] except last
                index_1 = find_all_sublists(input_ids,sub_squence_list[1])[:-1]
            else:
                # Default: Every [INST] and every [/INST]
                index_start = find_all_sublists(input_ids,sub_squence_list[0])
                index_1 = find_all_sublists(input_ids,sub_squence_list[1])
            track_index = index_start+index_1
            print(len(index_start),len(index_1))
            lat_list = [item for sublist in track_index for item in sublist]
            indexs.append(sorted(lat_list))
        train_token.append(tokenizer.encode(result))
    
    ## 将token序列化并保存
    os.makedirs(f'/root/project/few_vs_zero/data/data_token/{task}', exist_ok=True)
    with open(train_file, 'wb') as f:
        pickle.dump({"inputs": train_token,"indexs":indexs}, f)

## 加载模型并设置为训练模式
model = LlamaForCausalLM.from_pretrained(args.model, device_map='auto', torch_dtype=torch.bfloat16)
model.train()


## 损失函数
criterion = nn.CrossEntropyLoss(reduction="none")
out_data = [[0]*11008]*32

## 处理训练token并计算梯度
progress_bar = tqdm(total=len(train_token), leave=True, desc='Getting data')
for input_ids,index in zip(train_token, indexs):
    progress_bar.update(1)

    if len(input_ids)>1300:
        continue
    
    input_index = [i-1 for i in index]
    label_token = [input_ids[i] for i in index]

    input_ids = torch.tensor(input_ids,dtype=torch.int64).unsqueeze(0).to(device)
    label_token = torch.tensor(label_token,dtype=torch.int64).to(device)

    output = model(input_ids)
    loss1 = criterion(output.logits[0, input_index[:28], :], label_token[:28])
    loss2 = criterion(output.logits[0, input_index[28:], :], label_token[28:])

    # 计算平均损失
    loss = loss1.mean() + loss2.mean()
    model.zero_grad()

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None and "up_proj" in name:
            layer = int(name.split(".")[2])
            grad = torch.sum(param.grad,dim=1).cpu().tolist()
            out_data[layer] =  [abs(a) + b for a, b in zip(grad, out_data[layer])]

# 保存结果

folder_path = f"/root/project/few_vs_zero/data/matrix/{task}/{args.mod}"
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 {folder_path} 已创建。")
else:
    print(f"文件夹 {folder_path} 已经存在。")
with open(f"/root/project/few_vs_zero/data/matrix/{task}/{args.mod}/{args.shot}shot.json","w") as f:
    json.dump(out_data,f)
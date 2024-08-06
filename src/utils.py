import json
import random
import os

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

# path = "/home/songran/project/task-specific-neuron/natural-instructions-master/tasks/"
# file_list = ["task190_snli_classification.json","task227_clariq_classification.json","task075_squad1.1_answer_generation.json","task1645_medical_question_pair_dataset_text_classification.json",
# "task566_circa_classification.json","task379_agnews_topic_classification.json","task195_sentiment140_classification.json","task391_causal_relationship.json"]
# for i in file_list:
#     file_path = os.path.join(path,i)
#     print(i)
#     message_data = data_construct(file_path)
#     # print(message_data[0])
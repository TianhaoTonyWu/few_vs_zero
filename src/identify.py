import torch
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-t","--task", type=str, default="task274_overruling_legal_classification.json")
parser.add_argument("-md", "--mod", type=str, default="GV_trace-test")
parser.add_argument("-st","--shot",type=int, default=5)
args = parser.parse_args()

task = args.task.split("_")[0]
persent = 0.05

path = f"/home/wutianhao/project/few_vs_zero/data/matrix/{task}/{args.mod}/{args.shot}shot.json"

with open(path,"r") as f:
    data = json.load(f)

# 创建一个示例矩阵
matrix = torch.tensor(data)
layers,number = matrix.shape
flattened_matrix = matrix.view(-1)

# 计算需要的元素数量（最大的5%）
num_elements = flattened_matrix.numel()
top_k = int(num_elements *persent)

# 使用 torch.topk 找到最大的 5% 的值及其索引
top_values, top_indices = torch.topk(flattened_matrix, top_k)

# 将展平后的索引转换回原始矩阵的索引
rows = top_indices // matrix.size(1)
cols = top_indices % matrix.size(1)

# 将行列索引组合起来
top_indices_2d = torch.stack([
    top_indices // number,  # 行索引
    top_indices % number   # 列索引
], dim=1)
# original_indices = torch.stack((rows, cols), dim=1)

# 打印结果
print("最大的5%的值：", top_values)
print("这些值在原始矩阵中的索引：", top_indices_2d)

output = [[[] for i in range(layers)]]
for i in top_indices_2d:
    l,c = i
    output[0][l].append(c.item())

save_output = [[]]
q = 0
for j in output[0]:
    print(len(j))
    q+=len(j)
    # if len(j)==0:
    #     save_output[0].append(torch.tensor(0))
    # else:
    save_output[0].append(torch.tensor(j).type(torch.int64))
# print(output)
print(q)

# 保存结果
folder_path = f"/home/wutianhao/project/few_vs_zero/data/activation_mask/{task}"
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 {folder_path} 已创建。")
else:
    print(f"文件夹 {folder_path} 已经存在。")
torch.save(save_output,folder_path+f"/activation_{args.mod}_{task}_{args.shot}shot_pth")

import torch
import json
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-t","--task", type=str, default="task1447_drug_extraction_ade.json")
parser.add_argument("-md", "--mod", type=str, default="GV_trace-test")
parser.add_argument("-st","--shot",type=int, default=5)
args = parser.parse_args()

task = args.task.split("_")[0]
persent = 0.05

path = f"/root/project/few_vs_zero/data/matrix/{task}/{args.mod}/{args.shot}shot.json"

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


# 打印结果
print("最大的5%的值：", top_values)
print("这些值在原始矩阵中的索引：", top_indices_2d)

output = [[[] for i in range(layers)]]
for i in top_indices_2d:
    l,c = i
    output[0][l].append(c.item())

save_output = [[]]
for j in output[0]:
    # if len(j)==0:
    #     save_output[0].append(torch.tensor(0))
    # else:
    save_output[0].append(torch.tensor(j).type(torch.int64))



# 保存结果
folder_path = f"/root/project/few_vs_zero/data/activation_mask/{task}"
if not os.path.exists(folder_path):
    # 如果不存在，则创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 {folder_path} 已创建。")
else:
    print(f"文件夹 {folder_path} 已经存在。")
torch.save(save_output,folder_path+f"/activation_{args.mod}_{task}_{args.shot}shot_pth")


# Plot stats

# Initialize a list to store the count of activated neurons in each layer
activation_counts = [0] * layers

# Count the number of activations in each layer
for i in top_indices_2d:
    l, c = i
    activation_counts[l.item()] += 1

# Print the distribution of activations over the layers
print("Activation counts per layer:")
q = 0
for layer_index, count in enumerate(activation_counts):
    q += count
    print(f"Layer {layer_index}: {count} activations")

print("Total activation counts:", q)



### Plot the distribution as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(range(layers), activation_counts, color='blue')
plt.xlabel('Layer')
plt.ylabel('Number of Activated Neurons')
plt.title('Distribution of Activated Neurons (Top 5%)')
plt.grid(True)

plot_path = f"{folder_path}/activation_distribution_{args.mod}_{task}_{args.shot}shot.png"
plt.savefig(plot_path)



### Plot Heatmap of the whole matrix

# Reconstruct new matrix containing top 5%
new_matrix = torch.zeros_like(matrix)

for i in top_indices_2d:
    l, c = i
    new_matrix[l, c] = matrix[l, c]

def combine_neurons(matrix, block_size):
    layers, num_neurons = matrix.shape
    num_blocks = num_neurons // block_size
    remainder = num_neurons % block_size

    # Initialize a new matrix for combined blocks
    combined_matrix = torch.zeros((layers, num_blocks + (1 if remainder > 0 else 0)))
    
    for layer in range(layers):
        # Pad the last block if necessary
        padded_matrix = torch.cat([matrix[layer], torch.zeros(block_size - remainder)]) if remainder > 0 else matrix[layer]
        
        # Reshape and average the activations in blocks
        reshaped = padded_matrix.reshape(-1, block_size)
        combined_matrix[layer] = reshaped.mean(dim=1)
    
    return combined_matrix

# Combine every 100 adjacent neurons
block_size = 100
combined_matrix = combine_neurons(new_matrix, block_size)

# Convert the combined matrix to a numpy array for plotting
combined_matrix_np = combined_matrix.numpy()

plt.figure(figsize=(10, 6))
sns.heatmap(combined_matrix_np, cmap="inferno", cbar=False)
plt.xlabel('Neuron Block Index')
plt.ylabel('Layer')
plt.title('Heatmap of Neuron Activation Patterns Across Layers (Top 5%)')

heatmap_path = f"{folder_path}/activation_heatmap_{args.mod}_{task}_{args.shot}shot.png"
plt.savefig(heatmap_path)




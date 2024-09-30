from transformers import AutoModelForCausalLM
import torch
import baukit.nethook
import numpy as np
from utils import process_and_save_tokens
import matplotlib.pyplot as plt
import seaborn as sns


model_dir = "/root/autodl-tmp/model"

model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to('cuda')
for n, m in model.named_modules():
    print(n)



def get_activations(model, prompt): 

    model.eval()
    MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(32)]
    
    with torch.no_grad():
        with baukit.nethook.TraceDict(model, MLP_act) as ret:
            _ = model(prompt, output_hidden_states = True)
        MLP_act_value = [ret[act_value].output for act_value in MLP_act]
        return MLP_act_value
    
    
def act_llama(input_ids):
    mlp_act = get_activations(model,input_ids)
    mlp_act = [t.tolist() for t in mlp_act]
    return mlp_act

if __name__ == "__main__":
    # process tokens
    _, test_token, labels = process_and_save_tokens(
        task="task274_overruling_legal_classification.json",
        shot=0,
        tokenizer_name=model_dir
    )

    i = 0
    act = None
    # 每个prompt提取一次activation
    for input_ids in test_token[:1]:
        input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).to('cuda')
        res = act_llama(input_ids)
        print('###')
        print("prompt", i)
        print("shape of input tokens: ", input_ids.shape)
        squeezed = [item[0] for item in res]
        act = np.array(squeezed)
        print("shape of activation: ",act.shape)
        i += 1


    act = np.where(act > 0, 1, 0)
    ppt_act = np.sum(act, axis = 1)

    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    sns.heatmap(ppt_act, cmap='viridis')  # You can choose other colormaps too
    plt.title('Heatmap of the activation')
    plt.xlabel('Neuron Index')
    plt.ylabel('Layer Index')
    plt.savefig('activation.png')
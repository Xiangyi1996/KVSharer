#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')


# In[2]:
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List, Dict
from utils import read_json, write_json, get_keywords, get_alphabet_choice, remove_boxed, last_boxed_only_string
from agent import *
import logging
import pandas as pd
from ast import literal_eval
from collections import defaultdict
import sys
sys.path.insert(0, '/mnt/sh_flex_storage/home/xiangyiz/project/Symbolic-MoE/KVSharer')

# In[3]:


from llama_real_share.modeling_llama_kvsharer import LlamaForCausalLM


# ### Load Model
# 模型加载与校准数据准备

# In[31]:

# llama_path = 'meta-llama/Llama-2-7b-hf'
llama_path = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'


# In[28]:


download_dir="/mnt/sh_flex_storage/home/xiangyiz/project/Symbolic-MoE/saved_models"


# In[29]:


# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(llama_path, trust_remote_code=True)


# In[ ]:
llama = AutoModelForCausalLM.from_pretrained(llama_path, trust_remote_code=True).to('cuda')
# llama = LlamaForCausalLM.from_pretrained(llama_path, device_map='auto', trust_remote_code=True, local_files_only=True, cache_dir=download_dir)


# ### Load Calibration Dataset

# In[ ]:


wiki_data_path = './data/wiki_demo.txt'
with open(wiki_data_path, 'r') as f:
    wiki_data = f.readlines()
    f.close()


# In[ ]:


calibration_set = wiki_data[0:30] # 使用前30个样本进行校准


# ### Calculate the Euclidean Distance between any two layers of KV cache and sort them
# 
# 校准阶段：捕获每层的 KV 缓存
# - 对每个校准样本，执行前向传播并捕获每层的 KV 缓存（Key/Value）。
# - kv_cache_list 的结构为 [num_samples][num_layers][key, value]。

# In[ ]:


from tqdm import tqdm
import torch

kv_cache_share_layers_map = {i:i for i in range(len(llama.model.layers))}
kv_cache_list = [] # 存储所有样本的 KV 缓存
with torch.no_grad():
    for text in tqdm(calibration_set):  # 遍历校准数据集
        inp = tokenizer(text, return_tensors='pt', max_length=64, truncation=True)
        inp = inp.to('cuda')
        out = llama(**inp, kv_cache_share_layers_map=kv_cache_share_layers_map)
        past_key_values = out.past_key_values
        # 保存每层的 Key 和 Value 缓存
        kv_cache_list.append(past_key_values)


# In[ ]:


num_layers = len(kv_cache_list[0])
# 创建一个与 kv_cache_list 结构一致的列表 avg_past_key_values，用于存储每层 KV 缓存的平均值
# [num_samples][num_layers][key, value]
# avg_past_key_values：一个列表，每个元素是 (key_avg, value_avg)，初始值为全零张量，后续用于存储每层 Key 和 Value 缓存的平均值
# 创建与 kv_cache_list[0][i][0]（第 0 个样本的第 i 层 Key 缓存）形状和数据类型相同的全零张量。
# 示例：若 kv_cache_list[0][i][0] 的形状为 [1, 32, 64, 128]（batch_size=1, num_heads=32, seq_len=64, head_dim=128），则 torch.zeros_like(...) 会生成相同形状的零张量[[test_internlm.txt]]。
avg_past_key_values = [(torch.zeros_like(kv_cache_list[0][i][0]), torch.zeros_like(kv_cache_list[0][i][1])) for i in range(num_layers)]

for past_key_values in tqdm(kv_cache_list):
    for i, (key, value) in enumerate(past_key_values):
        try:
            #遍历所有校准样本的 KV 缓存（kv_cache_list），将每层的 Key 和 Value 分别累加到 avg_past_key_values 中
            avg_past_key_values[i] = (avg_past_key_values[i][0] + key, avg_past_key_values[i][1] + value)
        except:
            pass

#对累加后的 Key 和 Value 取平均值，得到每层的平均 KV 缓存
num_elements = len(kv_cache_list)
avg_past_key_values = [(key / num_elements, value / num_elements) for key, value in avg_past_key_values]


# ### 计算欧氏距离并排序
# - 展平操作: 将每层的 Key 和 Value 缓存展平为一维向量
# - 欧氏距离计算: 量化任意两层 KV 缓存的不相似性
# - 排序:按距离从大到小排列层对

# In[ ]:


import torch
import torch.nn.functional as F
import numpy as np

# 计算两个张量（如 KV 缓存）的余弦相似度
def compute_cosine_similarity(tensor1, tensor2):
    return F.cosine_similarity(tensor1.flatten(1), tensor2.flatten(1), dim=-1).mean().item()

# 计算两个张量之间的欧氏距离（值越大表示越不相似）
# 用于量化任意两层的不相似性（距离越大越优先替换)
def compute_euclidean_distance(tensor1, tensor2):
    return torch.norm(tensor1 - tensor2, p=2, dim=-1).mean().item()

num_layers = len(avg_past_key_values)
similarity_matrix = np.zeros((num_layers, num_layers))

# 构建相似度矩阵
# 遍历所有层对，计算每对的 Key 和 Value 的欧氏距离 ，并存储到 similarity_matrix 矩阵中。
# 仅计算上三角部分（i > j），避免重复计算。
for i in range(num_layers):
    for j in range(num_layers):
        if i > j:
            key_i, value_i = avg_past_key_values[i]
            key_j, value_j = avg_past_key_values[j]
            key_similarity = compute_euclidean_distance(key_i, key_j)
            value_similarity = compute_euclidean_distance(value_i, value_j)  
            similarity_matrix[i, j] = (key_similarity + value_similarity) / 2
        else:
            similarity_matrix[i, j] = np.nan


# In[ ]:


# 排序层对
# 将相似度矩阵展平并过滤无效值（NaN），按欧氏距离从大到小排序
flattened_values = similarity_matrix.flatten()
valid_indices = ~np.isnan(flattened_values)

# 将排序后的索引转换回原始矩阵的行列位置（即层对 (i, j)）
valid_values = flattened_values[valid_indices]
valid_flat_indices = np.where(valid_indices)[0]

sorted_valid_indices = np.argsort(valid_values)[::-1]
sorted_flat_indices = valid_flat_indices[sorted_valid_indices]

sorted_positions = np.unravel_index(sorted_flat_indices, similarity_matrix.shape)

pos_rank = []

for i in range(sorted_positions[0].shape[0]):
    pos = (sorted_positions[0][i], sorted_positions[1][i])
    pos_rank.append(pos)



# ### Initialize the Sharing Layers and THRESHOLD

# In[ ]:


SHARE_LAYERS = 4
THRESHOLD = 0.5


# In[ ]:


# 验证输出相似性 
# 对每个校准样本，比较替换策略下模型与原始模型的输出表示相似性（通过余弦相似度）
import numpy as np
def cal_last_hidden_sim(model1, model2, kv_cache_share_layers_map, tokenizer, sents):
    sim_ls = []
    for s in sents:
        encoded_inputs = tokenizer(s, max_length=64, truncation=True, return_tensors='pt')
        encoded_inputs.to('cuda')
        # 对每个校准样本，比较替换策略下模型与原始模型的输出表示相似性（通过余弦相似度）
        # 相似性计算 ：比较最后一层的隐藏状态（Hidden States）

        #model1 使用原始缓存
        with torch.no_grad():
            outputs1 = model1(**encoded_inputs, output_hidden_states=True, kv_cache_share_layers_map={i:i for i in range(len(model1.model.layers))})
        hidden_states1 = outputs1.hidden_states[-1] # (1, seq_len, hidden)

        #model2 使用共享策略（kv_cache_share_layers_map）
        with torch.no_grad():
            outputs2 = model2(**encoded_inputs, output_hidden_states=True, kv_cache_share_layers_map=kv_cache_share_layers_map)
        hidden_states2 = outputs2.hidden_states[-1] # (1, seq_len, hidden)
        sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)))
    sim_ls = [i.item() for i in sim_ls]
    print(sim_ls, np.mean(sim_ls))
    return np.mean(sim_ls)


# In[ ]:


# 按排序后的层对依次尝试替换
def re_map(kv_cache_share_layers_map):
    tmp_kv_cache_share_layers_map = {}
    for key, values in kv_cache_share_layers_map.items():
        if key == values:
            tmp_kv_cache_share_layers_map[key] = values
        else:
            tmp_kv_cache_share_layers_map[key] = tmp_kv_cache_share_layers_map[values]
    return tmp_kv_cache_share_layers_map


# ### Strategy Searching

# In[ ]:


from copy import deepcopy

# 初始化共享策略 ：kv_cache_share_layers_map 初始为 {i: i}，表示每层独立缓存
kv_cache_share_layers_map = {i:i for i in range(len(llama.model.layers))}

shared_lay = []
shared_num_layers = 0

# 遍历层对 ：按 pos_rank 降序排列的层对依次处理
for pair in tqdm(pos_rank):
    tmp_kv_cache_share_layers_map = deepcopy(kv_cache_share_layers_map)
    # 尝试将 pair[0] 的缓存替换为 pair[1] 的缓存
    if pair[0] < pair[1]:
        pair[0], pair[1] = pair[1], pair[0]
    if pair[0] in shared_lay:
        continue
    tmp_kv_cache_share_layers_map[pair[0]] = pair[1]
    # 调用 re_map 确保共享策略的一致性（避免链式映射）
    tmp_kv_cache_share_layers_map = re_map(tmp_kv_cache_share_layers_map)

    #通过 cal_last_hidden_sim 验证输出相似性
    sim_value = cal_last_hidden_sim(llama, llama, tmp_kv_cache_share_layers_map, tokenizer, calibration_set)

    #若相似性 > THRESHOLD，则保留替换
    if sim_value > THRESHOLD:
        kv_cache_share_layers_map = deepcopy(tmp_kv_cache_share_layers_map)
        shared_lay.append(pair[0])
        shared_num_layers += 1
    # 替换层数达到 SHARE_LAYERS（如 8 层）后停止
    if shared_num_layers >= SHARE_LAYERS:
        break


# In[ ]:


print(kv_cache_share_layers_map)


# ### Inference with KVSharer

# In[ ]:


def get_model_responses(model, tokenizer, sent, kv_cache_share_layers_map=None):
    inputs = tokenizer(sent, return_tensors='pt')
    inputs = inputs.to('cuda')
    pred = model.generate(**inputs, kv_cache_share_layers_map=kv_cache_share_layers_map, max_new_tokens=256, repetition_penalty=1.1)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

def get_valid_responses(row, answer_columns):
    responses = {}
    for col in answer_columns:
        if pd.notna(row[col]) and pd.notnull(row[col]):
            responses[col] = row[col]
    return responses

# In[ ]:

# sent = 'Hello, what is your name'
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aggregator',
        type=str,
        default="QwenR1"
    )
    parser.add_argument(
        '--task',
        type=str,
        default="GPQA"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1
    )

    parser.add_argument("--no_use_chunk_prefill", action="store_true")

    parser.add_argument("--max_num_seqs", type=int, default=128)

    parser.add_argument("--suffix", type=str, default='GPQA')

    parser.add_argument("--kv_cache_dtype", type=str, default="bf16")

    parser.add_argument("--quantization", type=str, default="None")

    return parser.parse_args()

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
if __name__ == "__main__":

    args = parse_args()
    seed_everything(args.seed)
    start_time = time.time()
    num_choice = 10 if args.task == "MMLU_Pro" else 4
    test_samples = read_json(f"./test_data/{args.task}_test.json")
    round_zero_df = pd.read_csv(f"./skills_all/skills_{args.suffix}/{args.task}/round0_seed{args.seed}.csv")
    # round_zero_df = pd.read_csv(f"./skills_all/skills/{args.task}/round0_seed{args.seed}.csv")
    answer_columns = [col for col in round_zero_df.columns if 'answer_' in col]
    num_correct = 0
    for i, row in round_zero_df.iterrows():
        gt = str(row['gold_answer'])
        valid_responses = get_valid_responses(row, answer_columns)
        preds = []
        for r in list(valid_responses.values()):
            if args.task in ["MATH_Hard", "AIME24", "OmniMATH"]:
                pred = remove_boxed(last_boxed_only_string(r))
            else:
                pred = get_alphabet_choice(r, num_choice=num_choice)
            preds.append(pred)
        maj = Counter(preds).most_common(1)[0][0]

        if is_math_equiv(maj, gt):
            num_correct += 1
    acc = round(num_correct / round_zero_df.shape[0] * 100, 2)
    print(f"Initial accuracy with the 3 experts: ", acc)

    agg_prompts = []
    for i, row in round_zero_df.iterrows():
        q = row['question']
        agg_prompt = (f"You have been provided with a set of responses from various open-source models to the latest user query. "
                      f"Your task is to synthesize these responses into a single, high-quality response. "
                      f"It is crucial to critically evaluate the information provided in these responses, "
                      f"recognizing that some of it may be biased or incorrect. "
                      f"Your response should not simply replicate the given answers but should offer a refined, "
                      f"accurate, and comprehensive reply to the instruction. "
                      f"Ensure your response is well-structured, coherent, and adheres"
                      f"to the highest standards of accuracy and reliability. "
                      f"Responses from models:\n\n")

        valid_responses = get_valid_responses(row, answer_columns)
        valid_responses = list(valid_responses.values())
        for idx, res in enumerate(valid_responses):
            res  = res.split("</think>")[-1]
            agg_prompt += f"### Model {idx+1}'s response:\n{res}\n\n"

        if args.task in ["MATH_Hard", "AIME24", "OmniMATH"]:
            agg_prompt += (f"Question: {q}\n"
                           f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
                           f"where X is the final answer, at the end of your response."
                        )
        else:
            agg_prompt += (f"Question: {q}\n"
                           f"Provide your step-by-step reasoning first, and then print \"The answer is (X)\", "
                           f"where X is the answer choice (one capital letter), at the end of your response."
                    )
        agg_prompts.append(agg_prompt)

    start_time = time.time()
    round_zero_df = round_zero_df.loc[:, ['question', 'gold_answer', 'keywords', 'solvers']] # get rid of all prev answers
    if args.no_use_chunk_prefill:
        use_chunk_prefill = False
    else:
        use_chunk_prefill = True
    
    result = get_model_responses(llama, tokenizer, agg_prompts, kv_cache_share_layers_map=kv_cache_share_layers_map)
    
    # result = get_model_responses(
    #     args.aggregator,
    #     agg_prompts,
    #     args.gpus,
    # )

    num_correct = 0
    for r, ts in zip(result, test_samples):
        gt = ts['gold_answer']
        if args.task in ["MATH_Hard", "AIME24", "OmniMATH"]:
            pred = remove_boxed(last_boxed_only_string(r))
        else:
            pred = get_alphabet_choice(r, num_choice=num_choice)
        if is_math_equiv(pred, gt):
            num_correct += 1
    acc = round(num_correct / len(test_samples) * 100, 2)
    print(f"acc: {acc} | dataset: {args.task} | aggregator: {args.aggregator} | seed: {args.seed}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Aggregator latency: {duration}")

    create_folder(f"./skills_all/skills_{args.suffix}/{args.task}")
    write_json(
        result,
        f"./skills_all/skills_{args.suffix}/{args.task}/fixed_{args.aggregator}_round1_seed{args.seed}_{acc}.json",
    )


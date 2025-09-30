#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import time
import logging
import argparse
import random
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from ast import literal_eval
from torch.utils.data import DataLoader
from copy import deepcopy
import torch.nn.functional as F
# Local imports
from utils import read_json, write_json, get_keywords, get_alphabet_choice, remove_boxed, last_boxed_only_string
from agent import *
from E2E.run.utils import load_dataset_local
from datasets import Dataset, DatasetDict

os.environ['HF_ENDPOINT']='https://hf-mirror.com'  # 使用HuggingFace镜像站
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

total_output_tokens = 0
total_time = 0
memory_usage = []

class KVCacheManager:
    """KV缓存管理类，封装KV共享核心逻辑"""
    
    def __init__(
            self,
            model,
            tokenizer: AutoTokenizer,
            calibration_set: List[str],
            threshold: float = 0.95,
            max_shared_layers: int = 1,
            device: str = "cuda",
            share_similar_layers: bool = False,
            use_euclidean_distance: bool = True
    ):
        """
        初始化KVCacheManager
        
        Args:
            share_similar_layers (bool): 
                - True: 共享相似的层 
                - False: 共享不相似的层 (KVSharer原始论文方法，推荐)
            use_euclidean_distance (bool):
                - True: 使用欧氏距离计算层间差异 (KVSharer原始论文方法，推荐)
                - False: 使用余弦相似度计算层间差异
        """
        self.model = model
        self.tokenizer = tokenizer
        self.calibration_set = calibration_set
        self.threshold = threshold
        self.max_shared_layers = max_shared_layers
        self.num_layers = len(model.model.layers)
        self.kv_cache_list = []  # 存储校准后的KV缓存
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.share_similar_layers = share_similar_layers
        self.use_euclidean_distance = use_euclidean_distance
        
        print(f"🔧 KVSharer配置:")
        print(f"   - share_similar_layers={share_similar_layers}")
        print(f"   - use_euclidean_distance={use_euclidean_distance}")
        
        if use_euclidean_distance:
            print("✅ 距离计算: 欧氏距离 (KVSharer原始论文方法)")
        else:
            print("🔄 距离计算: 余弦相似度")
            
        if share_similar_layers:
            print("⚠️  层选择策略: 共享相似/近距离层")
        else:
            print("✅ 层选择策略: 共享不相似/远距离层 (KVSharer原始论文方法)")

    
    def calibrate(self, max_length: int = 64):
        """校准阶段：捕获每层KV缓存"""
        logger.info("开始KVSharer校准...")
        # 初始化：每层使用自己的缓存（无共享）
        kv_cache_share_layers_map = {i: i for i in range(self.num_layers)}
        
        with torch.no_grad():
            for text in tqdm(self.calibration_set, desc="校准进度"):
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # 关键：正确配置输出参数
                outputs = self.model(
                    **inputs,
                    kv_cache_share_layers_map=kv_cache_share_layers_map,
                    use_cache=True,              # 必须开启缓存
                    return_dict=True,            # 返回字典格式
                    output_hidden_states=False   # 不需要所有隐藏状态
                )
                
                # 正确获取KV缓存
                if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                    self.kv_cache_list.append(outputs.past_key_values)
                else:
                    logger.warning(f"校准样本 '{text[:50]}...' 没有生成KV缓存")
        
        logger.info(f"校准完成，收集了 {len(self.kv_cache_list)} 个样本的KV缓存")
    
    def average_kv_cache(self, ):
        num_layers = len(self.kv_cache_list[0])
        # 创建一个与 kv_cache_list 结构一致的列表 avg_past_key_values，用于存储每层 KV 缓存的平均值
        # [num_samples][num_layers][key, value]
        # avg_past_key_values：一个列表，每个元素是 (key_avg, value_avg)，初始值为全零张量，后续用于存储每层 Key 和 Value 缓存的平均值
        # 创建与 kv_cache_list[0][i][0]（第 0 个样本的第 i 层 Key 缓存）形状和数据类型相同的全零张量。
        # 示例：若 kv_cache_list[0][i][0] 的形状为 [1, 32, 64, 128]（batch_size=1, num_heads=32, seq_len=64, head_dim=128），则 torch.zeros_like(...) 会生成相同形状的零张量[[test_internlm.txt]]。
        avg_past_key_values = [(torch.zeros_like(self.kv_cache_list[0][i][0]), torch.zeros_like(self.kv_cache_list[0][i][1])) for
                               i in range(num_layers)]
        
        for past_key_values in tqdm(self.kv_cache_list):
            for i, (key, value) in enumerate(past_key_values):
                try:
                    # 遍历所有校准样本的 KV 缓存（kv_cache_list），将每层的 Key 和 Value 分别累加到 avg_past_key_values 中
                    avg_past_key_values[i] = (avg_past_key_values[i][0] + key, avg_past_key_values[i][1] + value)
                except:
                    pass
        
        # 对累加后的 Key 和 Value 取平均值，得到每层的平均 KV 缓存
        num_elements = len(self.kv_cache_list)
        self.avg_past_key_values = [(key / num_elements, value / num_elements) for key, value in avg_past_key_values]
    
    # 计算两个张量（如 KV 缓存）的余弦相似度
    def compute_cosine_similarity(self, tensor1, tensor2):
        similarity = F.cosine_similarity(tensor1.flatten(1), tensor2.flatten(1), dim=-1).mean().item()
        # 处理 NaN 值：如果相似度是 NaN，返回 0（表示不相似）
        if torch.isnan(torch.tensor(similarity)):
            print(f"⚠️ 检测到 NaN 相似度，设为 0")
            return 0.0
        return similarity
    
    # 计算两个张量之间的欧氏距离（值越大表示越不相似）
    # 用于量化任意两层的不相似性（距离越大越优先替换)
    def compute_euclidean_distance(self, tensor1, tensor2):
        return torch.norm(tensor1 - tensor2, p=2, dim=-1).mean().item()
    
    def analyze_kv_similarity(self) -> np.ndarray:
        """分析各层KV缓存相似度/距离，生成矩阵"""
        logger.info("分析层间KV缓存相似度...")
        self.average_kv_cache()
        num_layers = len(self.avg_past_key_values)
        
        # 创建矩阵来存储层间关系
        distance_matrix = np.zeros((num_layers, num_layers))
        
        if self.use_euclidean_distance:
            print("📏 计算欧氏距离 (KVSharer原始论文方法)...")
            # 计算每层之间的欧氏距离（值越大越不相似）
            for i in range(num_layers):
                for j in range(i + 1, num_layers):  # 只计算上三角
                    key_i, value_i = self.avg_past_key_values[i]
                    key_j, value_j = self.avg_past_key_values[j]
                    
                    # 使用欧氏距离（值越大越不相似）
                    key_distance = self.compute_euclidean_distance(key_i, key_j)
                    value_distance = self.compute_euclidean_distance(value_i, value_j)
                    avg_distance = (key_distance + value_distance) / 2
                    
                    # 处理可能的 NaN 或 inf 值
                    if torch.isnan(torch.tensor(avg_distance)) or torch.isinf(torch.tensor(avg_distance)):
                        print(f"⚠️ 检测到无效距离值 (层 {i}-{j})，设为最大距离")
                        avg_distance = 1e6  # 设为一个很大的距离值
                    
                    # 存储距离
                    distance_matrix[i, j] = avg_distance
                    distance_matrix[j, i] = avg_distance  # 对称矩阵
        else:
            print("📐 计算余弦相似度...")
            # 计算每层之间的余弦相似度（值越大越相似）
            for i in range(num_layers):
                for j in range(i + 1, num_layers):  # 只计算上三角
                    key_i, value_i = self.avg_past_key_values[i]
                    key_j, value_j = self.avg_past_key_values[j]
                    
                    # 使用余弦相似度（值越大越相似）
                    key_similarity = self.compute_cosine_similarity(key_i, key_j)
                    value_similarity = self.compute_cosine_similarity(value_i, value_j)
                    avg_similarity = (key_similarity + value_similarity) / 2
                    
                    # 存储相似度
                    distance_matrix[i, j] = avg_similarity
                    distance_matrix[j, i] = avg_similarity  # 对称矩阵
        
        return distance_matrix
    
    def sort_similarity(self, share_similar_layers=False):
        """
        按距离/相似度排序层对来决定共享策略
        
        Args:
            share_similar_layers (bool): 
                - True: 共享相似/近距离的层
                - False: 共享不相似/远距离的层 (KVSharer原始论文方法)
        """
        distance_matrix = self.analyze_kv_similarity()
        
        # 获取上三角的层对
        layer_pairs = []
        for i in range(distance_matrix.shape[0]):
            for j in range(i + 1, distance_matrix.shape[1]):
                layer_pairs.append((i, j, distance_matrix[i, j]))
        
        if self.use_euclidean_distance:
            # 欧氏距离：值越大越不相似
            if share_similar_layers:
                # 共享相似层：选择距离小的（升序排列）
                layer_pairs.sort(key=lambda x: x[2], reverse=False)
                print("🔄 策略: 共享距离近的层 (欧氏距离小)")
                if len(layer_pairs) > 0:
                    min_dist = layer_pairs[0][2]
                    max_dist = layer_pairs[-1][2]
                    print(f"   - 距离范围: {min_dist:.4f} -> {max_dist:.4f}")
            else:
                # 共享不相似层：选择距离大的（降序排列） - KVSharer原理
                layer_pairs.sort(key=lambda x: x[2], reverse=True)
                print("✅ 策略: 共享距离远的层 (KVSharer原始论文方法)")
                if len(layer_pairs) > 0:
                    max_dist = layer_pairs[0][2]
                    min_dist = layer_pairs[-1][2]
                    print(f"   - 距离范围: {max_dist:.4f} -> {min_dist:.4f}")
        else:
            # 余弦相似度：值越大越相似
            if share_similar_layers:
                # 共享相似层：选择相似度高的（降序排列）
                layer_pairs.sort(key=lambda x: x[2], reverse=True)
                print("🔄 策略: 共享相似度高的层")
                if len(layer_pairs) > 0:
                    max_sim = layer_pairs[0][2]
                    min_sim = layer_pairs[-1][2]
                    print(f"   - 相似度范围: {max_sim:.4f} -> {min_sim:.4f}")
            else:
                # 共享不相似层：选择相似度低的（升序排列）
                layer_pairs.sort(key=lambda x: x[2], reverse=False)
                print("✅ 策略: 共享相似度低的层")
                if len(layer_pairs) > 0:
                    min_sim = layer_pairs[0][2]
                    max_sim = layer_pairs[-1][2]
                    print(f"   - 相似度范围: {min_sim:.4f} -> {max_sim:.4f}")
        
        return [(pair[0], pair[1]) for pair in layer_pairs]

    import numpy as np
    def cal_last_hidden_sim(self, model1, model2, kv_cache_share_layers_map, tokenizer, sents):
        # 验证输出相似性
        # 对每个校准样本，比较替换策略下模型与原始模型的输出表示相似性（通过余弦相似度）
        sim_ls = []
        for s in sents:
            encoded_inputs = tokenizer(s, max_length=64, truncation=True, return_tensors='pt')
            encoded_inputs.to('cuda')
            # 对每个校准样本，比较替换策略下模型与原始模型的输出表示相似性（通过余弦相似度）
            # 相似性计算 ：比较最后一层的隐藏状态（Hidden States）
            
            # model1 使用原始缓存
            with torch.no_grad():
                outputs1 = model1(**encoded_inputs, output_hidden_states=True,
                                  kv_cache_share_layers_map={i: i for i in range(len(model1.model.layers))})
            hidden_states1 = outputs1.hidden_states[-1]  # (1, seq_len, hidden)
            
            # model2 使用共享策略（kv_cache_share_layers_map）
            with torch.no_grad():
                outputs2 = model2(**encoded_inputs, output_hidden_states=True,
                                  kv_cache_share_layers_map=kv_cache_share_layers_map)
            hidden_states2 = outputs2.hidden_states[-1]  # (1, seq_len, hidden)
            sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0),
                                                  hidden_states2.squeeze(0).flatten().unsqueeze(0)))
        sim_ls = [i.item() for i in sim_ls]
        # print(sim_ls, np.mean(sim_ls))
        return np.mean(sim_ls)
    
    def re_map(self, kv_cache_share_layers_map):
        # 按排序后的层对依次尝试替换
        tmp_kv_cache_share_layers_map = {}
        for key, values in kv_cache_share_layers_map.items():
            if key == values:
                tmp_kv_cache_share_layers_map[key] = values
            else:
                tmp_kv_cache_share_layers_map[key] = tmp_kv_cache_share_layers_map[values]
        return tmp_kv_cache_share_layers_map
    
    def build_sharing_strategy(self, threshold, max_shared_layers) -> Dict[int, int]:
        shared_lay = []
        shared_num_layers = 0
        total_layers = self.model.config.num_hidden_layers
        compression_ratio = 0.0
        kv_cache_share_layers_map = {i: i for i in range(len(self.model.model.layers))}
        # 遍历层对：按指定策略排列的层对依次处理
        pos_rank = self.sort_similarity(self.share_similar_layers)
        for i, pair in enumerate(tqdm(pos_rank)):
            tmp_kv_cache_share_layers_map = deepcopy(kv_cache_share_layers_map)
            # 尝试将 pair[0] 的缓存替换为 pair[1] 的缓存
            # 确保 pair[0] > pair[1]，即较大索引的层使用较小索引层的缓存
            if pair[0] < pair[1]:
                pair = (pair[1], pair[0])  # 创建新的元组而不是修改原元组
            if pair[0] in shared_lay:
                continue
            tmp_kv_cache_share_layers_map[pair[0]] = pair[1]
            # 调用 re_map 确保共享策略的一致性（避免链式映射）
            tmp_kv_cache_share_layers_map = self.re_map(tmp_kv_cache_share_layers_map)
            
            # 通过 cal_last_hidden_sim 验证输出相似性
            sim_value = self.cal_last_hidden_sim(self.model, self.model, tmp_kv_cache_share_layers_map, self.tokenizer, self.calibration_set)
            
            # 若相似性 > THRESHOLD，则保留替换
            if sim_value > threshold:
                kv_cache_share_layers_map = deepcopy(tmp_kv_cache_share_layers_map)
                shared_lay.append(pair[0])
                shared_num_layers += 1
                compression_ratio = shared_num_layers / total_layers
                
                # 实时报告压缩率和相似度
                print(f"Step {i + 1}: Layer {pair[0]} -> {pair[1]} | "
                      f"Shared: {shared_num_layers}/{total_layers} | "
                      f"Compression: {compression_ratio:.2%} | "
                      f"Similarity: {sim_value:.4f}")
                print(kv_cache_share_layers_map)
            # 替换层数达到 SHARE_LAYERS（如 8 层）后停止
            if shared_num_layers >= max_shared_layers:
                print(
                    f"Reached target compression: {compression_ratio:.2%} ({shared_num_layers}/{total_layers} layers)")
                break
        
        # 计算并报告最终压缩率
        final_compression = len(shared_lay) / total_layers
        print(f"\nStrategy built with {len(shared_lay)}/{total_layers} layers shared")
        print(f"Final compression ratio: {final_compression:.2%} ({final_compression * 100:.1f}%)")
        self.final_compression = final_compression
        
        # 关键：检查压缩率是否过高
        if final_compression > 0.25:  # KVSharer论文推荐不超过25%
            print(
                f"⚠️ WARNING: Compression ratio {final_compression:.2%} exceeds recommended 25% - may cause accuracy drop")
            
        print('1: ', kv_cache_share_layers_map)
        return kv_cache_share_layers_map
    
class ModelInference:
    def __init__(self, args, device: str = "cuda"):
        self.model_name = args.aggregator
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.download_dir = args.download_dir
        self.tokenizer = None
        self.model = None
        self.math500_enabled = args.math500_enabled
        self.attn_implementation = args.attn_implementation

        # 策略配置
        self.strategy = args.strategy  # "kvsharer", "adaskip", "fusion", "baseline"
        self.kvsharer_enabled = self.strategy in ["kvsharer"]
        self.adaskip_enabled = self.strategy in ["adaskip"]

        # KVSharer参数
        self.threshold = args.threshold
        self.max_shared_layers = args.max_shared_layers

        # AdaSkip参数  
        self.skip_sub_layer_num = args.skip_sub_layer_num
        self.max_gen = args.max_gen

        # 融合模式下的共享层和可跳过层
        self.kv_cache_share_layers_map = None
        self.skip_candidate_layers = None
        
        self.common_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "attn_implementation": self.attn_implementation,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
            "cache_dir": "/mnt/sh_flex_storage/home/xiangyiz/project/Symbolic-MoE/saved_models"
        }

        # 加载校准数据

        # 加载模型和构建共享策略
        if self.strategy == "fusion":
            # 1. 先加载支持KVSharer的模型
            self.load_calibration_data()
            self._load_model()
            # 2. 构建可跳过层集合（未被share的层）
            num_layers = self.model.config.num_hidden_layers
            all_layers = set(range(num_layers))
            shared_layers = set()
            for k, v in self.kv_cache_share_layers_map.items():
                if k!=v:
                    shared_layers.add(k)
                    shared_layers.add(v)
            self.skip_candidate_layers = sorted(list(all_layers - shared_layers))
            print(f"[Fusion] KVSharer共享层: {sorted(list(shared_layers))}")
            print(f"[Fusion] AdaSkip可跳过层: {self.skip_candidate_layers}")
            # 关键注释：skip_candidate_layers会在generate_response_adaskip中传递给模型，确保AdaSkip只在未share层上跳过
        
        else:
            self._load_model()

        print(f"🎯 策略配置: {self.strategy}")
        print(f"   - KVSharer: {self.kvsharer_enabled}")
        print(f"   - AdaSkip: {self.adaskip_enabled}")
        

        
    def _read_jsonl(self, ) -> List[Dict]:
        """Read a JSONL file where each line is a separate JSON object."""
        file_path = f"../test_data/math500_test.jsonl"
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if len(data) >= 200:
                    break
                if line.strip():  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON at line: {line.strip()}")
                        raise e
        return data
    
    def calculate_model_weight_memory(self, model):
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 ** 3)
        
    def load_calibration_data(self):
        if self.math500_enabled:
            print("math500 is enabled in calibration")
            math_500_samples = self._read_jsonl()
            calibration_prompts = [
                f"Question: {sample['problem']}\n"
                "Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
                "where X is the final answer, at the end of your response."
                for sample in math_500_samples
            ]
            self.calibration_data = calibration_prompts
        else:
            self.wiki_data_path = './data/wiki_demo.txt'
            with open(self.wiki_data_path, 'r') as f:
                wiki_data = f.readlines()
                f.close()

            self.calibration_data = wiki_data[0:60]  # 使用前30个样本进行校准
        
    
    def _load_kvsharer_model(self, model_id):
        # 加载支持KVSharer的模型
        self.load_calibration_data()
        
        from llama_real_share.modeling_llama_kvsharer_4470 import LlamaForCausalLM as LlamaR1ForCausalLM
        from qwen_real_share.qwenR1_test import Qwen2ForCausalLM
        
        MODEL_CLASS_MAP = {
            'QwenR1': Qwen2ForCausalLM,
            'LlamaR1': LlamaR1ForCausalLM,
        }
        model_class = MODEL_CLASS_MAP.get(self.model_name)
        if not model_class:
            raise ValueError(f"Unsupported model name for KV sharing: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=True,
            padding=True,
            truncation=True,
        )
        
        self.model = model_class.from_pretrained(
            model_id,
            **self.common_kwargs
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 初始化KV缓存管理器
        self.kv_manager = KVCacheManager(
            model=self.model,
            tokenizer=self.tokenizer,
            calibration_set=self.calibration_data,
            threshold=self.threshold,
            max_shared_layers=self.max_shared_layers
        )
        
        # 执行校准和策略构建
        self.kv_manager.calibrate()
        self.kv_cache_share_layers_map = self.kv_manager.build_sharing_strategy(self.threshold, self.max_shared_layers)
    
    def _load_fusion_model(self, model_id):
        try:
            from E2E.model.llama_ADA import LlamaForCausalLM, MyLlamaConfig
            from E2E.model.qwen2_ADA import Qwen2ForCausalLM, MyQwenR1Config
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # 使用自定义的MyLlamaConfig，传入SKIP_SUB_LAYER_NUM参数
            if self.model_name == 'LlamaR1':
                config = MyLlamaConfig.from_pretrained(model_id, SKIP_SUB_LAYER_NUM=self.skip_sub_layer_num)
                self.model = LlamaForCausalLM.from_pretrained(model_id, config=config, **self.common_kwargs).to(self.device)
            elif self.model_name == 'QwenR1':
                config = MyQwenR1Config.from_pretrained(model_id, SKIP_SUB_LAYER_NUM=self.skip_sub_layer_num)
                self.model = Qwen2ForCausalLM.from_pretrained(model_id, config=config, **self.common_kwargs).to(self.device)
            else:
                raise ValueError(f"Unsupported model for AdaSkip: {self.model_name}")
                
            # 初始化KV缓存管理器
            self.kv_manager = KVCacheManager(
                model=self.model,
                tokenizer=self.tokenizer,
                calibration_set=self.calibration_data,
                threshold=self.threshold,
                max_shared_layers=self.max_shared_layers
            )
            
            # 执行校准和策略构建
            self.load_calibration_data()
            self.kv_manager.calibrate()
            self.kv_cache_share_layers_map = self.kv_manager.build_sharing_strategy(self.threshold,
                                                                                    self.max_shared_layers)
            # AdaSkip的重要性收集和跳层决策完全在模型内部自动处理
            # 无需外部管理器，模型在推理过程中会自动：
            # 1. 收集重要性信息 (attn_prefill_sim_diffreq_avg等)
            # 2. 在diff_req_count>=20时确定FIX_SKIP_*_LAYER_ID
            # 3. 在decode过程中动态决定额外跳过的层
            print(f"🚀 AdaSkip模型已加载，自适应跳层将在推理过程中自动进行")
            print(f"   - skip_sub_layer_num={self.skip_sub_layer_num}")
            print(f"   - 模型将在处理20个不同请求后确定固定跳层策略")
            
            logger.info("AdaSkip模型加载成功")
            model_weight_memory = self.calculate_model_weight_memory(self.model)
            logger.info(f"Model Weight Memory: {model_weight_memory:.2f} GB")
        
        except Exception as e:
            logger.error(f"Error loading AdaSkip model: {str(e)}")
            raise
        
        self.model = self.model.eval()
    def _load_model(self) -> None:
        """加载标准模型（支持KVSharer或baseline）"""
        try:
            model_id = agent_map.get(self.model_name)
            logger.info(f"Loading model {self.model_name}")

            if self.kvsharer_enabled:
                print(f"🚀 kvsharer enabled")
                self._load_kvsharer_model(model_id)
            elif self.adaskip_enabled:
                print(f"🚀 adaskip enabled")
                self._load_adaskip_model(model_id)
            elif self.strategy == "fusion":
                print(f"🚀 fusion enabled")
                self._load_fusion_model(model_id)
            else:
                # 加载标准模型（baseline）
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **self.common_kwargs
                ).to(self.device)
            
            logger.info("Model loaded successfully")
            model_weight_memory = self.calculate_model_weight_memory(self.model)
            logger.info(f"Model Weight Memory: {model_weight_memory:.2f} GB")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _load_adaskip_model(self, model_id):
        try:
            from E2E.model.llama_ADA import LlamaForCausalLM, MyLlamaConfig
            from E2E.model.qwen2_ADA import Qwen2ForCausalLM, MyQwenR1Config
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # 使用自定义的MyLlamaConfig，传入SKIP_SUB_LAYER_NUM参数
            if self.model_name == 'LlamaR1':
                config = MyLlamaConfig.from_pretrained(model_id, SKIP_SUB_LAYER_NUM=self.skip_sub_layer_num)
                self.model = LlamaForCausalLM.from_pretrained(model_id, config=config, **self.common_kwargs).to(self.device)
            elif self.model_name == 'QwenR1':
                config = MyQwenR1Config.from_pretrained(model_id, SKIP_SUB_LAYER_NUM=self.skip_sub_layer_num)
                self.model = Qwen2ForCausalLM.from_pretrained(model_id, config=config, **self.common_kwargs).to(self.device)
            else:
                raise ValueError(f"Unsupported model for AdaSkip: {self.model_name}")
                
            # AdaSkip的重要性收集和跳层决策完全在模型内部自动处理
            # 无需外部管理器，模型在推理过程中会自动：
            # 1. 收集重要性信息 (attn_prefill_sim_diffreq_avg等)
            # 2. 在diff_req_count>=20时确定FIX_SKIP_*_LAYER_ID
            # 3. 在decode过程中动态决定额外跳过的层
            print(f"🚀 AdaSkip模型已加载，自适应跳层将在推理过程中自动进行")
            print(f"   - skip_sub_layer_num={self.skip_sub_layer_num}")
            print(f"   - 模型将在处理20个不同请求后确定固定跳层策略")
            
            logger.info("AdaSkip模型加载成功")
            model_weight_memory = self.calculate_model_weight_memory(self.model)
            logger.info(f"Model Weight Memory: {model_weight_memory:.2f} GB")
        
        except Exception as e:
            logger.error(f"Error loading AdaSkip model: {str(e)}")
            raise
        
        self.model = self.model.eval()
    
    def load_warmup_data(self):
        if self.math500_enabled:
            print("load math500 warmup dataset")
            math_500_samples = self._read_jsonl()
            calibration_prompts = [
                f"Question: {sample['problem']}\n"
                "Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
                "where X is the final answer, at the end of your response."
                for sample in math_500_samples
            ]
            prompt_all = calibration_prompts
        else:
            prompt_all = []
            print("load LongBench warmup dataset")
            warmup_dataset = "qasper"  # "Anyone is OK"
            warmup_data = load_dataset_local(warmup_dataset)
            warmup_data_all = [data_sample for data_sample in warmup_data]
            dataset2prompt = json.load(open(f"/mnt/sh_flex_storage/home/xiangyiz/project/Symbolic-MoE/AdaSkip/config/dataset2prompt.json", "r"))
            prompt_format = dataset2prompt[warmup_dataset]
            for json_obj in tqdm(warmup_data_all):
                prompt = prompt_format.format(**json_obj) # 格式化提示
                prompt_all.append(prompt)
        return prompt_all
    
    def generate_response_adaskip(self, agg_prompts, ):
        """
        支持AdaSkip和融合（fusion）模式。
        融合模式下，先应用KVSharer共享层，再在未share层上做AdaSkip。
        skip_candidate_layers和kv_cache_share_layers_map会传递给模型，确保AdaSkip只在未share层上跳过。
        """
        global total_time

        if self.strategy == "fusion":
            skip_candidate_layers = self.skip_candidate_layers
            kv_cache_share_layers_map = self.kv_cache_share_layers_map
        else:
            skip_candidate_layers = None
            kv_cache_share_layers_map = None

        # Warmup 预填充阶段
        warmup_data_all = self.load_warmup_data()
        warm_up_count = 0
        for prompt in tqdm(warmup_data_all):
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            max_length = 31500
            if len(tokenized_prompt) > max_length:
                print(f"warmup data is exceed {max_length} tokens")
                half = int(max_length / 2)
                prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + self.tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True)
            input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.device)
            context_length = input.input_ids.shape[-1]
            # 运行模型但不保存输出，仅用于收集子层重要性信息
            output = self.model.generate(
                **input,
                max_new_tokens=1,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                kv_cache_share_layers_map=kv_cache_share_layers_map,
            )[0]
            
            warm_up_count += 1
            if warm_up_count >= 20:
                break

        batch_start_time = time.time()

        dataloader = DataLoader(agg_prompts, batch_size=1)
        result = []
        for prompt in tqdm(dataloader, desc="Evaluating"):
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            max_length = 31500
            if len(tokenized_prompt) > max_length:
                print(f"warmup data is exceed {max_length} tokens")
                half = int(max_length / 2)
                prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True)
            input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.device)
            context_length = input.input_ids.shape[-1]

            max_gen = self.max_gen
            output = self.model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                min_length=context_length + 1,
                temperature=1.0,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.encode("\n", add_special_tokens=False)[-1]],
                kv_cache_share_layers_map=kv_cache_share_layers_map,
            )[0]
            
            pred = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
            # logger.info("=========Pred==========")
            # logger.info(pred)
            # logger.info("=========Close==========")
            result.append(pred)

            global total_output_tokens
            global memory_usage

            output_length = len(output[context_length:])
            print(f"input_length: {context_length}")
            print(f"output_length: {output_length}")
            kv_cache_memory = self.calculate_kvcache_memory(self.model, context_length, output_length, batch_size=1)
            print(f"KV Cache Memory: {kv_cache_memory:.2f} GB")
            total_output_tokens += output_length
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                memory_usage.append({
                    "kv_cache_memory": kv_cache_memory,
                    "torch_peak_memory": peak_memory,
                })

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        total_time += batch_duration
        return result
    
    def generate_response(
            self,
            prompt: str,
            batch_size: int,
            max_new_tokens: int = 32768,
            repetition_penalty: float = 1.1
    ) -> str:
        """生成回复（仅用于KVSharer和baseline策略）"""
        global total_time
        return self._generate_with_kvsharer(prompt, batch_size, max_new_tokens, repetition_penalty)
    
    def _generate_with_kvsharer(self, prompt: str, batch_size: int, max_new_tokens: int, repetition_penalty: float) -> str:
        """使用KVSharer或baseline策略生成"""
        global total_time

        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(self.device)

        # 记录单次请求时间
        batch_start_time = time.time()

        with torch.no_grad():
            if self.kvsharer_enabled:
                outputs = self.model.generate(
                    **inputs,
                    kv_cache_share_layers_map=self.kv_cache_share_layers_map,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
            else:
                # Baseline策略
                outputs = self.model.generate(
                    **inputs,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
                    
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        total_time += batch_duration
        
        self.analyze_memory(inputs, outputs, batch_size)
        
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        logger.info("=========Pred==========")
        logger.info(pred)
        logger.info("=========Close==========")
        
        return pred
    
    
    def calculate_kvcache_memory(self, model, input_length, output_length, batch_size):
        """计算不同策略下的KV缓存内存占用"""
        num_hidden_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        bytes_per_element = 2  # FP16 每个元素占 2 字节
        
        if self.strategy == "kvsharer":
            # KVSharer策略：减少共享层的内存占用
            total_bytes = 2 * batch_size * (input_length + output_length) * \
                         (num_hidden_layers * (1 - self.kv_manager.final_compression)) * \
                         hidden_dim * bytes_per_element
        elif self.strategy == "adaskip":
            # AdaSkip策略：跳过的层不产生KV缓存
            effective_layers = num_hidden_layers - self.skip_sub_layer_num
            print(f"AdaSkip有效层数: {effective_layers}/{num_hidden_layers}")
            total_bytes = 2 * batch_size * (input_length + output_length) * \
                         effective_layers * hidden_dim * bytes_per_element
        elif self.strategy == "fusion":
            # 融合策略：同时考虑KVSharer和AdaSkip的内存节省
            effective_layers = num_hidden_layers - self.skip_sub_layer_num
            if hasattr(self, 'kv_manager'):
                compression = self.kv_manager.final_compression
            else:
                compression = 0.0
            total_bytes = 2 * batch_size * (input_length + output_length) * \
                         (effective_layers * (1 - compression)) * hidden_dim * bytes_per_element
        else:
            # Baseline策略：所有层都产生KV缓存
            total_bytes = 2 * batch_size * (input_length + output_length) * \
                         num_hidden_layers * hidden_dim * bytes_per_element
        
        return total_bytes / (1024 ** 3)  # 转换为 GB
    
    def analyze_memory(self, inputs, outputs, batch_size=1):
        # 计算 KV Cache 内存
        global total_output_tokens
        global memory_usage
        input_length = inputs["input_ids"].shape[1]
        
        output_length = outputs.shape[1] - input_length
        
        print(f"input_length: {input_length}")
        print(f"output_length: {output_length}")
        
        kv_cache_memory = self.calculate_kvcache_memory(self.model, input_length, output_length, batch_size)
        # print(f"KV Cache Memory: {kv_cache_memory:.2f} GB")
        
        # 记录总 token 数
        output_tokens = sum([output.shape[0] for output in outputs])
        total_output_tokens += output_length
        
        # 记录显存占用（GB）
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # print(f"Peak Memory Usage: {peak_memory:.2f} GB")
            
            memory_usage.append({
                "kv_cache_memory": kv_cache_memory,
                "torch_peak_memory": peak_memory,
            })
        

class ResultEvaluator:
    @staticmethod
    def evaluate_math_response(pred_all, gt_all, task_type: str, num_choice: int) -> bool:
        """Evaluate mathematical response equivalence"""
        num_correct = 0
        for r, ts in zip(pred_all, gt_all):
            gt = ts['gold_answer']
            if task_type in ["MATH_Hard", "AIME24", "OmniMATH"]:
                pred = remove_boxed(last_boxed_only_string(r))
            else:
                pred = get_alphabet_choice(r, num_choice=num_choice)
            if is_math_equiv(pred, gt):
                num_correct += 1
        return num_correct


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_tester = ModelInference(args)
        self.results = []
        self.num_correct = 0
    
    def get_valid_responses(self, row, answer_columns):
        responses = {}
        for col in answer_columns:
            if pd.notna(row[col]) and pd.notnull(row[col]):
                responses[col] = row[col]
        return responses
    
    def log_performance(self):
        tokens_per_second = total_output_tokens / total_time if total_time > 0 else 0
        avg_output_len = total_output_tokens / len(memory_usage) if len(memory_usage) > 0 else 0
        
        logger.info("\n=== Performance Statistics ===")
        logger.info(f"Total Tokens: {total_output_tokens}")
        logger.info(f"Tokens Per Second: {tokens_per_second:.2f} tokens/s")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info(f"Average seq len: {avg_output_len:.2f}")
        
        if torch.cuda.is_available():
            avg_kv_cache = sum([m["kv_cache_memory"] for m in memory_usage]) / len(memory_usage)
            avg_torch_peak = sum([m["torch_peak_memory"] for m in memory_usage]) / len(memory_usage)
            
            logger.info(f"Peak Memory Usage: {avg_torch_peak:.2f} GB")
            logger.info(f"Avg KV Cache Memory: {avg_kv_cache:.2f} GB")
    
    
    def run_evaluation(self) -> None:
        """Run model evaluation pipeline"""
        self.test_samples = read_json(f"./test_data/{self.args.task}_test.json")
        
        
        
        logger.info(f"Starting evaluation on {len(self.test_samples)} samples")
        num_choice = 10 if self.args.task == "MMLU_Pro" else 4
        if self.args.task in ["MATH_Hard", "AIME24", "OmniMATH"]:
            round_zero_df = pd.read_csv(f"./skills_all/skills_{self.args.suffix}/{self.args.task}/round0_seed{self.args.seed}_k3.csv")
        else:
            round_zero_df = pd.read_csv(f"./skills_all/skills_{self.args.suffix}/{self.args.task}/round0_seed{self.args.seed}.csv")
        if self.args.num_infer != None:
            round_zero_df = round_zero_df[:self.args.num_infer]
        answer_columns = [col for col in round_zero_df.columns if 'answer_' in col]
        num_correct = 0
        for i, row in round_zero_df.iterrows():
            gt = str(row['gold_answer'])
            valid_responses = self.get_valid_responses(row, answer_columns)
            preds = []
            for r in list(valid_responses.values()):
                if self.args.task in ["MATH_Hard", "AIME24", "OmniMATH"]:
                    pred = remove_boxed(last_boxed_only_string(r))
                else:
                    pred = get_alphabet_choice(r, num_choice=num_choice)
                preds.append(pred)
            maj = Counter(preds).most_common(1)[0][0]
            
            if is_math_equiv(maj, gt):
                num_correct += 1
        acc = round(num_correct / round_zero_df.shape[0] * 100, 2)
        logger.info(f"Initial accuracy with the 3 experts: {acc}" )
        
        agg_prompts = []
        for i, row in round_zero_df.iterrows():
            q = row['question']
            agg_prompt = (
                f"You have been provided with a set of responses from various open-source models to the latest user query. "
                f"Your task is to synthesize these responses into a single, high-quality response. "
                f"It is crucial to critically evaluate the information provided in these responses, "
                f"recognizing that some of it may be biased or incorrect. "
                f"Your response should not simply replicate the given answers but should offer a refined, "
                f"accurate, and comprehensive reply to the instruction. "
                f"Ensure your response is well-structured, coherent, and adheres"
                f"to the highest standards of accuracy and reliability. "
                f"Responses from models:\n\n")
            
            valid_responses = self.get_valid_responses(row, answer_columns)
            valid_responses = list(valid_responses.values())
            for idx, res in enumerate(valid_responses):
                res = res.split("</think>")[-1]
                agg_prompt += f"### Model {idx + 1}'s response:\n{res}\n\n"
            
            if self.args.task in ["MATH_Hard", "AIME24", "OmniMATH"]:
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
        round_zero_df = round_zero_df.loc[:,
                        ['question', 'gold_answer', 'keywords', 'solvers']]  # get rid of all prev answers
        
        # 关键设计决策：AdaSkip vs 其他策略的不同处理方式
        result = []

        if self.args.strategy == "adaskip" or self.args.strategy == "fusion":
            # AdaSkip策略需要特殊处理：
            # 1. 不使用DataLoader的批处理（需要完整prompt列表进行预热）
            # 2. 需要先进行预热阶段收集重要性信息
            # 3. 然后对每个prompt单独进行目标推理
            result = self.model_tester.generate_response_adaskip(agg_prompts)
        else:
            # KVSharer、baseline或fusion策略使用标准批处理：
            # 1. 使用DataLoader进行批处理
            # 2. 每个batch调用generate_response方法
            # 3. 策略在模型加载时已经确定（KVSharer）或在推理时决定（baseline）
            dataloader = DataLoader(agg_prompts, batch_size=self.args.batch_size)
            for sample in tqdm(dataloader, desc="推理中"):
                assert all(isinstance(p, str) for p in sample), "批次中的元素应该是字符串"
                response = self.model_tester.generate_response(sample,
                                                               self.args.batch_size,
                                                               max_new_tokens=self.args.max_new_tokens)
                result.extend(response)
        
        self.log_performance()
        
        self.num_correct = ResultEvaluator.evaluate_math_response(
            result, self.test_samples, self.args.task, num_choice
        )
        
        duration = time.time() - start_time
        self._log_statistics(duration, len(round_zero_df))
        self._save_results(result)
    
    
    def _log_statistics(self, duration: float, total_samples: int) -> None:
        """Log evaluation statistics"""
        self.acc = round(self.num_correct / total_samples * 100, 2)
        logger.info(f"Accuracy: {self.acc}% | Dataset: {self.args.task}")
        logger.info(f"Total time: {duration:.2f}s | Avg time per sample: {duration / total_samples:.2f}s")
    
    def _save_results(self, result) -> None:
        """Save evaluation results to file"""
        output_dir = f"./skills_all/skills_{self.args.suffix}/{self.args.task}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/fixed_{self.args.aggregator}_round1_seed{self.args.seed}_{self.acc}.json"
        
        write_json(result, output_file)
        logger.info(f"Results saved to {output_file}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--task", type=str,
                        choices=["MATH_Hard", "AIME24", "OmniMATH", "GPQA"], default="GPQA",
                        help="评估任务类型")
    parser.add_argument("--aggregator", type=str, default="QwenR1",
                        help="聚合方法名称")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子")
    parser.add_argument("--suffix", type=str, default="GPQA",
                        help="输出目录后缀")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_infer", type=int, default=None)
    parser.add_argument("--download_dir", type=str, default="/mnt/sh_flex_storage/home/xiangyiz/project/Symbolic-MoE/saved_models")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--math500_enabled", type=str2bool, default=False)

    # 策略配置 - 四种选择
    parser.add_argument("--strategy", type=str, 
                        choices=["baseline", "kvsharer", "adaskip", "fusion"], 
                        default="baseline",
                        help="优化策略: baseline(无优化), kvsharer(仅KV共享), adaskip(仅子层跳过), fusion(融合)")
    
    # KVSharer参数
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="KVSharer相似度阈值")
    parser.add_argument("--max_shared_layers", type=int, default=1,
                        help="KVSharer最大共享层数")

    # AdaSkip参数
    parser.add_argument('--skip_sub_layer_num', type=int, default=8, 
                        help='AdaSkip跳过的子层数量')
    parser.add_argument('--max_gen', type=int, default=31000,
                        help='AdaSkip最大生成长度')

    # 向后兼容的参数（已弃用，但保留以免破坏现有脚本）
    parser.add_argument("--kvsharer_enabled", type=str2bool, default=False,
                        help="已弃用，请使用--strategy kvsharer")
    parser.add_argument("--adaskip_enabled", type=str2bool, default=False,
                        help="已弃用，请使用--strategy adaskip")

    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    
    return parser.parse_args()




def main():
    """主函数"""
    try:
        # 设置随机种子
        args = parse_args()
        
        # 向后兼容性处理
        if args.kvsharer_enabled and args.strategy == "baseline":
            args.strategy = "kvsharer"
            logger.warning("检测到--kvsharer_enabled=True，自动设置strategy=kvsharer")
        
        if args.adaskip_enabled and args.strategy == "baseline":
            args.strategy = "adaskip"
            logger.warning("检测到--adaskip_enabled=True，自动设置strategy=adaskip")
        
        if args.kvsharer_enabled and args.adaskip_enabled:
            args.strategy = "fusion"
            logger.warning("检测到两种优化都启用，自动设置strategy=fusion")
        
        logger.info("参数配置: " + str(args))
        logger.info(f"🎯 使用策略: {args.strategy}")
        
        # 策略验证
        if args.strategy == "fusion":
            logger.warning("⚠️ 融合策略仍在开发中，某些功能可能不稳定")
        
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # 运行实验
        runner = ExperimentRunner(args)
        runner.run_evaluation()
        logger.info("参数配置: " + str(args))

    except KeyboardInterrupt:
        logger.info("用户中断评估")
    except Exception as e:
        logger.error(f"评估过程中出错: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
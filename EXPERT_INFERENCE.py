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
import contextlib
import gc
import os
import subprocess

# Local imports
from utils import read_json, write_json, get_keywords, get_alphabet_choice, remove_boxed, last_boxed_only_string
from agent import *
import pprint
# from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3


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
            threshold: float = 0.5,
            max_shared_layers: int = 4,
            device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.calibration_set = calibration_set
        self.threshold = threshold
        self.max_shared_layers = max_shared_layers
        self.num_layers = len(model.model.layers)
        self.kv_cache_list = []  # 存储校准后的KV缓存
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    def calibrate(self, max_length: int = 512):
        """校准阶段：捕获每层KV缓存"""
        logger.info("Calibration...")
        kv_cache_share_layers_map = {i: i for i in range(self.num_layers)}
        
        with torch.no_grad():
            for text in tqdm(self.calibration_set, desc="Calibration"):
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                outputs = self.model(
                    **inputs,
                    kv_cache_share_layers_map=kv_cache_share_layers_map,
                    output_hidden_states=True
                )
                self.kv_cache_list.append(outputs.past_key_values)
    
    def average_kv_cache(self, ):
        num_layers = len(self.kv_cache_list[0])
        # 创建一个与 kv_cache_list 结构一致的列表 avg_past_key_values，用于存储每层 KV 缓存的平均值
        # [num_samples][num_layers][key, value]
        # avg_past_key_values：一个列表，每个元素是 (key_avg, value_avg)，初始值为全零张量，后续用于存储每层 Key 和 Value 缓存的平均值
        # 创建与 kv_cache_list[0][i][0]（第 0 个样本的第 i 层 Key 缓存）形状和数据类型相同的全零张量。
        # 示例：若 kv_cache_list[0][i][0] 的形状为 [1, 32, 64, 128]（batch_size=1, num_heads=32, seq_len=64, head_dim=128），则 torch.zeros_like(...) 会生成相同形状的零张量[[test_internlm.txt]]。
        avg_past_key_values = [
            (torch.zeros_like(self.kv_cache_list[0][i][0]), torch.zeros_like(self.kv_cache_list[0][i][1])) for
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
        return F.cosine_similarity(tensor1.flatten(1), tensor2.flatten(1), dim=-1).mean().item()
    
    # 计算两个张量之间的欧氏距离（值越大表示越不相似）
    # 用于量化任意两层的不相似性（距离越大越优先替换)
    def compute_euclidean_distance(self, tensor1, tensor2):
        return torch.norm(tensor1 - tensor2, p=2, dim=-1).mean().item()
    
    def analyze_kv_similarity(self) -> np.ndarray:
        """分析各层KV缓存相似度，生成距离矩阵"""
        logger.info("Analyze kV Cache similarity between layers and generate distance matrix...")
        self.average_kv_cache()
        num_layers = len(self.avg_past_key_values)
        
        # 初始化距离矩阵
        distance_matrix = np.zeros((num_layers, num_layers))
        
        # 计算每层之间的余弦相似度
        # 构建相似度矩阵
        # 遍历所有层对，计算每对的 Key 和 Value 的欧氏距离 ，并存储到 similarity_matrix 矩阵中。
        # 仅计算上三角部分（i > j），避免重复计算。
        for i in range(num_layers):
            for j in range(num_layers):
                if i > j:
                    key_i, value_i = self.avg_past_key_values[i]
                    key_j, value_j = self.avg_past_key_values[j]
                    key_similarity = self.compute_euclidean_distance(key_i, key_j)
                    value_similarity = self.compute_euclidean_distance(value_i, value_j)
                    distance_matrix[i, j] = (key_similarity + value_similarity) / 2
                else:
                    distance_matrix[i, j] = np.nan
        
        return distance_matrix
    
    def sort_distance(self):
        # 排序层对
        # 将相似度矩阵展平并过滤无效值（NaN），按欧氏距离从大到小排序
        distance_matrix = self.analyze_kv_similarity()
        flattened_values = distance_matrix.flatten()
        valid_indices = ~np.isnan(flattened_values)
        
        # 将排序后的索引转换回原始矩阵的行列位置（即层对 (i, j)）
        valid_values = flattened_values[valid_indices]
        valid_flat_indices = np.where(valid_indices)[0]
        
        sorted_valid_indices = np.argsort(valid_values)[::-1]
        sorted_flat_indices = valid_flat_indices[sorted_valid_indices]
        
        sorted_positions = np.unravel_index(sorted_flat_indices, distance_matrix.shape)
        
        pos_rank = []
        
        for i in range(sorted_positions[0].shape[0]):
            pos = (sorted_positions[0][i], sorted_positions[1][i])
            pos_rank.append(pos)
        
        return pos_rank
    
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
        # 遍历层对 ：按 pos_rank 降序排列的层对依次处理
        pos_rank = self.sort_distance()
        for i, pair in enumerate(tqdm(pos_rank)):
            tmp_kv_cache_share_layers_map = deepcopy(kv_cache_share_layers_map)
            # 尝试将 pair[0] 的缓存替换为 pair[1] 的缓存
            if pair[0] < pair[1]:
                pair[0], pair[1] = pair[1], pair[0]
            if pair[0] in shared_lay:
                continue
            tmp_kv_cache_share_layers_map[pair[0]] = pair[1]
            # 调用 re_map 确保共享策略的一致性（避免链式映射）
            tmp_kv_cache_share_layers_map = self.re_map(tmp_kv_cache_share_layers_map)
            
            # 通过 cal_last_hidden_sim 验证输出相似性
            sim_value = self.cal_last_hidden_sim(self.model, self.model, tmp_kv_cache_share_layers_map, self.tokenizer,
                                                 self.calibration_set)
            
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
                 # model_name: str, download_dir: str, use_kvsharer: bool = True, THRESHOLD: float = 0.5,
                 # MAX_SHARED_LAYERS: int = 1, device: str = "cuda", deer_enabled: bool = False, deer_threshold: float = 0.95,
                 # deer_think_ratio: float = 0.8, deer_max_len: int = 8192,
                 # deer_answer_len: int = 64
        self.model_name = args.aggregator
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.download_dir = args.download_dir
        self.tokenizer = None
        self.model = None
        self.math500_enabled = args.math500_enabled
        

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
        self.dtype = args.torch_dtype
        self.attn_implementation = args.attn_implementation
        if self.dtype == 'fp16':
            torch_dtype = torch.float16
        elif self.dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif self.dtype == 'fp32':
            torch_dtype = torch.float32
        else:
            logger.warning(f"Unknown dtype: {self.dtype}, defaulting to float16")
            torch_dtype = torch.float16
            
        self.common_kwargs = {
             "torch_dtype": torch_dtype,
             "trust_remote_code": True,
             "attn_implementation": self.attn_implementation,
             "low_cpu_mem_usage": True,
             # "use_cache": True,
             "local_files_only": True,
             "cache_dir": self.download_dir
         }
        
        print(f"🎯 策略配置: {self.strategy}")


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
            from E2E.run.utils import load_dataset_local
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
    
    def load_calibration_data(self):
        if self.args.math500_enabled:
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
        
    def _load_rkv_model(self, model_id):
        ## Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # ====== build compression config ======
        compression_config = {
            "method": self.args.method,
            "method_config": {
                "budget": self.args.kv_budget,
                "window_size": self.args.window_size,
                "mix_lambda": self.args.mix_lambda,
                "retain_ratio": self.args.retain_ratio,
                "retain_direction": self.args.retain_direction,
                "first_tokens": self.args.first_tokens,
            },
            "compression": None,
            "update_kv": self.args.update_kv
        }
        model_config = {
            "divide_method": self.args.divide_method,
            "divide_length": self.args.divide_length,
            "compression_content": self.args.compression_content,
        }
        # apply monkey patch
        if self.args.method.lower() != "fullkv":
            if "llama" in model_id.lower():
                replace_llama(compression_config)
            elif "qwen3" in model_id.lower():
                replace_qwen3(compression_config)
            elif "qwen" in model_id.lower():
                replace_qwen2(compression_config)
            else:
                raise ValueError(f"Unsupported model: {model_id}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            attn_implementation=self.args.attn_implementation,
        )
        self.model.eval()
        self.model.config.update(model_config)
        if self.args.method.lower() != "fullkv":
            self.model.newline_token_ids = [
                self.tokenizer.encode("\n")[-1],
                self.tokenizer.encode(".\n")[-1],
                self.tokenizer.encode(")\n")[-1],
                self.tokenizer.encode("\n\n")[-1],
                self.tokenizer.encode(".\n\n")[-1],
                self.tokenizer.encode(")\n\n")[-1],
            ]
            
            self.model.after_think_token_ids = [
                self.tokenizer.encode("</think>")[-1],
            ]
            
    def _load_model(self, model_name) -> None:
        """Load model and tokenizer with proper device placement"""
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
            elif self.strategy == "rkv":
                self._load_rkv_model(model_id)
            else:
                # 加载标准模型（baseline）
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **self.common_kwargs
                ).to(self.device)
                
            model_id = agent_map.get(model_name)
            logger.info(f"Loading model {model_name}")
            logger.info(f"{model_id} model loaded successfully")
            model_weight_memory = self.calculate_model_weight_memory(self.model)
            logger.info(f"Model Weight Memory: {model_weight_memory:.2f} GB")
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        
    def calculate_model_weight_memory(self, model):
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 ** 3)  # 转换为 GB [[3]](https://zhuanlan.zhihu.com/p/96840298 )
    
    def get_model_responses(
            self,
            agent_name,
            prompts,
            args
    ):
        model_id = agent_map.get(agent_name)
        logger.info(f"Loading model {agent_name}")
        
        self._load_model(model_name=model_id)
        
        # 批量生成配置
        dataloader = DataLoader(prompts, batch_size=args.batch_size)
        
        results = []
        total_output_tokens = 0
        total_input_len = 0
        total_output_len = 0
        total_time = 0
        memory_usage = []
        
        for batch in tqdm(dataloader, desc="Processing Batches"):
            # 确保 batch 是字符串列表
            assert all(isinstance(p, str) for p in batch), "Batch 中的所有元素必须是字符串"
            
            # 使用 Tokenizer 并启用填充和截断
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                # padding=True,
                # truncation=True,
                # max_length=131072,
                # add_special_tokens=True
            ).to("cuda")
            
            input_length = inputs["input_ids"].shape[1]
            
            # 记录单次请求时间
            batch_start_time = time.time()
            
            # 生成响应
            with torch.no_grad():
                if self.kvsharer_enabled:
                    outputs = self.model.generate(
                        **inputs,
                        kv_cache_share_layers_map=self.kv_cache_share_layers_map,
                        max_new_tokens=self.args.max_gen,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        repetition_penalty=1.1
                    )[0]
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_gen,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        repetition_penalty=1.1
                    )[0]
            
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            total_time += batch_duration
            
            # 解码生成结果
            context_length = inputs.input_ids.shape[-1]
            pred = self.tokenizer.decode(outputs[context_length:], skip_special_tokens=True)
            # logger.info("=========Pred==========")
            # logger.info(pred)
            # logger.info("=========Close==========")
            results.extend(pred)
            
            # 计算 KV Cache 内存
            output_length = len(outputs[context_length:])
            print(f"input_length: {input_length}")
            print(f"output_length: {output_length}")
            
            kv_cache_memory = self.calculate_kvcache_memory(self.model, input_length, output_length,
                                                                         batch_size=args.batch_size)
            print(f"KV Cache Memory: {kv_cache_memory:.2f} GB")
            # 记录总 token 数
            
            total_input_len += input_length
            total_output_len += output_length
            
            # 记录显存占用（GB）
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"Peak Memory Usage: {peak_memory:.2f} GB")
                memory_usage.append({
                    "kv_cache_memory": kv_cache_memory,
                    "torch_peak_memory": peak_memory,
                })
        
        # 输出 Profiler 统计结果
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        
        tokens_per_second = total_output_len / total_time if total_time > 0 else 0
        avg_input_len = total_input_len / len(memory_usage) if len(memory_usage) > 0 else 0
        avg_output_len = total_output_len / len(memory_usage) if len(memory_usage) > 0 else 0
        
        print("\n=== Performance Statistics ===")
        print(f"Total Samples: {len(memory_usage)}")
        print(f"Total Tokens: {total_output_tokens}")
        print(f"Tokens Per Second: {tokens_per_second:.2f} tokens/s")
        # print(f"Total Time: {total_time:.2f} seconds | Avg time per sample: {total_time / len(memory_usage):.2f}s")
        print(f"Average input len: {avg_input_len:.2f}")
        print(f"Average output len: {avg_output_len:.2f}")
        
        if torch.cuda.is_available():
            # avg_model_weight = max([m["model_weight_memory"] for m in memory_usage])
            avg_kv_cache = sum([m["kv_cache_memory"] for m in memory_usage]) / len(memory_usage)
            avg_torch_peak = sum([m["torch_peak_memory"] for m in memory_usage]) / len(memory_usage)
            
            print(f"Peak Memory Usage: {avg_torch_peak:.2f} GB")
            print(f"Avg KV Cache Memory: {avg_kv_cache:.2f} GB")
            # print(f"Avg Model weight Memory: {avg_model_weight:.2f} GB")
        
        return results
        gc.collect()
        torch.cuda.empty_cache()
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        time.sleep(10)
        subprocess.run(["nvidia-smi"], check=True)
        return results
    
    def generate_response_adaskip(self, agent_name, agg_prompts, args):
        """
        支持AdaSkip和融合（fusion）模式。
        融合模式下，先应用KVSharer共享层，再在未share层上做AdaSkip。
        skip_candidate_layers和kv_cache_share_layers_map会传递给模型，确保AdaSkip只在未share层上跳过。
        """
        global total_time
        global total_output_tokens
        global memory_usage

        model_id = agent_map.get(self.model_name)
        logger.info(f"Loading model {self.model_name}")

        self._load_adaskip_model(model_id)

        # Warmup 预填充阶段
        warmup_data_all = self.load_warmup_data()
        warm_up_count = 0
        for prompt in tqdm(warmup_data_all):
            tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            max_length = 31500
            if len(tokenized_prompt) > max_length:
                print(f"warmup data is exceed {max_length} tokens")
                half = int(max_length / 2)
                prompt = self.tokenizer.decode(tokenized_prompt[:half],
                                                            skip_special_tokens=True) + self.model_tester.tokenizer.decode(
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
                kv_cache_share_layers_map=self.kv_cache_share_layers_map,
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
                prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + self.tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True)
            input = self.tokenizer(prompt, truncation=False, return_tensors="pt").to(self.device)
            context_length = input.input_ids.shape[-1]

            max_gen = self.max_gen

            # output = self.model.generate(
            #     **input,
            #     max_new_tokens=max_gen,
            #     do_sample=True,
            #     temperature=0.7,
            #     top_p=0.9,
            #     eos_token_id=self.tokenizer.eos_token_id,
            #     pad_token_id=self.tokenizer.pad_token_id,
            #     repetition_penalty=1.1,
            # )[0]
            output = self.model.generate(
                **input,
                max_new_tokens=max_gen,
                do_sample=False,
                temperature=1,
                num_beams=1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]

            pred = self.tokenizer.decode(output[context_length:], skip_special_tokens=True)
            logger.info("=========Pred==========")
            logger.info(pred)
            logger.info("=========Close==========")
            result.append(pred)


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

        tokens_per_second = total_output_tokens / total_time if total_time > 0 else 0
        avg_output_len = total_output_tokens / len(memory_usage) if len(memory_usage) > 0 else 0

        print("\n=== Performance Statistics ===")
        print(f"Total Samples: {len(memory_usage)}")
        print(f"Total Tokens: {total_output_tokens}")
        print(f"Tokens Per Second: {tokens_per_second:.2f} tokens/s")
        print(f"Total Time: {total_time:.2f} seconds | Avg time per sample: {total_time / len(memory_usage):.2f}s")
        print(f"Average output len: {avg_output_len:.2f}")

        if torch.cuda.is_available():
            avg_kv_cache = sum([m["kv_cache_memory"] for m in memory_usage]) / len(memory_usage)
            avg_torch_peak = sum([m["torch_peak_memory"] for m in memory_usage]) / len(memory_usage)

            print(f"Peak Memory Usage: {avg_torch_peak:.2f} GB")
            print(f"Avg KV Cache Memory: {avg_kv_cache:.2f} GB")
        
        # gc.collect()
        torch.cuda.empty_cache()
        # import torch.distributed as dist
        # if dist.is_initialized():
        #     dist.destroy_process_group()
        # with contextlib.suppress(AssertionError):
        #     torch.distributed.destroy_process_group()
        # time.sleep(10)
        # subprocess.run(["nvidia-smi"], check=True)
        return result

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_valid_responses(self, row, answer_columns):
        responses = {}
        for col in answer_columns:
            if pd.notna(row[col]) and pd.notnull(row[col]):
                responses[col] = row[col]
        return responses
    
    def create_folder(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")
        else:
            print(f"Folder '{folder_path}' already exists.")
    

    
    def calculate_model_weight_memory(self, model):
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 ** 3)  # 转换为 GB [[3]](https://zhuanlan.zhihu.com/p/96840298 )
    
    
    
    
    
    def analyze_memory(self, inputs=None, outputs=None, total_tokens=0, batch_size=1, model=None):
        global total_output_tokens
        global memory_usage
        
        input_length = inputs
        output_length = outputs
        kv_cache_memory = self.calculate_kvcache_memory(model, input_length, output_length, batch_size)
        print(f"KV Cache Memory: {kv_cache_memory:.2f} GB")
        # 记录总 token 数
        total_output_tokens += output_length
        
        # 记录显存占用（GB）
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # print(f"Peak Memory Usage: {peak_memory:.2f} GB")
            
            memory_usage.append({
                "kv_cache_memory": kv_cache_memory,
                "torch_peak_memory": peak_memory,
            })
        
    def generate_response_rkv(self, agent_name, prompts, batch_size=1):
        result = []
        global total_time
        self.model_tester._load_model(model_name=agent_name)
        
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_start_time = time.time()
            
            batch_prompts = prompts[i: i + batch_size]
            tokenized_prompts = self.model_tester.tokenizer(
                batch_prompts,
                padding="longest",
                return_tensors="pt",
                add_special_tokens=True,
            ).to(self.device)
            
            prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()
            
            output = self.model_tester.model.generate(
                **tokenized_prompts,
                max_new_tokens=self.args.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
            
            batch_token_stats = []
            for j in range(output.size(0)):
                total_tokens = int((output[j] != self.model_tester.tokenizer.pad_token_id).sum().item())
                
                prefill = prefill_lengths[j]
                output_tokens = total_tokens - prefill
                
                batch_token_stats.append(
                    {
                        "sample_idx": i + j,
                        "prefill_tokens": prefill,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }
                )
                self.analyze_memory(inputs=prefill, outputs=output_tokens, total_tokens=total_tokens,
                                    batch_size=batch_size, model=self.model_tester.model)
            
            batch_outputs = self.model_tester.tokenizer.batch_decode(
                [output[j][prefill_lengths[j]:] for j in range(output.size(0))],
                skip_special_tokens=True,
            )
            
            result.extend(batch_outputs)
            
            torch.cuda.empty_cache()
            
            # for j in range(len(batch_outputs)):
            #     sample_idx = batch_token_stats[j]["sample_idx"]
            #     test_data[sample_idx]["prompt"] = batch_prompts[j]
            #     test_data[sample_idx]["output"] = batch_outputs[j]
            #     test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
            #     test_data[sample_idx]["output_tokens"] = batch_token_stats[j]["output_tokens"]
            #     test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
            #     test_data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]
            #
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            total_time += batch_duration
        
        return result
    
    def process_questions_and_answers(self, df, task, gpus=1):
        # Convert string representations of lists to actual lists
        df['solvers'] = df['solvers'].apply(literal_eval)
        
        # Create a mapping for each solver to their questions and tracking answer counts
        solver_questions = defaultdict(list)
        question_answer_counts = defaultdict(lambda: defaultdict(int))
        
        # Map questions to solvers and count occurrences
        for idx, row in df.iterrows():
            question = row['question']
            if task in ["MATH_Hard", "AIME24", "OmniMATH"]:
                prompt = (
                    f"Question: {question}\n"
                    f"Provide your step-by-step reasoning first, and then print \"The answer is \\boxed{{X}}\", "
                    f"where X is the final answer, at the end of your response."
                )
            else:
                prompt = (
                    f"Question: {question}\n"
                    f"Provide your step-by-step reasoning first, and then print \"The answer is (X)\", "
                    f"where X is the answer choice (one capital letter), at the end of your response."
                )
            for solver in row['solvers']:
                solver_questions[solver].append((idx, prompt))
                question_answer_counts[idx][solver] += 1
        
        # Process each solver's questions and update DataFrame
        for solver, questions in solver_questions.items():
            # Get question IDs and texts separately
            q_ids, q_texts = zip(*questions)
            
            # Get model responses
            if self.args.rkv_enabled:
                responses = self.generate_response_rkv(solver, q_texts, batch_size=1)
            elif self.args.strategy == 'adaskip':
                responses = self.model_tester.generate_response_adaskip(solver, q_texts, self.args)
            else:
                responses = self.model_tester.get_model_responses(solver, q_texts, self.args)
            
            # Add responses to DataFrame
            for q_id, response in zip(q_ids, responses):
                answer_num = question_answer_counts[q_id][solver]
                question_answer_counts[q_id][solver] -= 1
                col_name = f"{solver}_answer_{answer_num}"
                df.loc[q_id, col_name] = response
        
        return df
    

    
    def run_evaluation(self) -> None:
        """Run model evaluation pipeline"""
        df = pd.read_csv(
            f"./skills_all/skills/{self.args.task}/test_samples_with_keywords_and_solvers_seed{self.args.seed}.csv")
        self.create_folder(f"./skills_all/skills_{self.args.suffix}/{self.args.task}")
        
        logger.info(f"Starting Expert Inference on {len(df)} samples")
        
        if self.args.num_infer != None:
            df = df[:self.args.num_infer]
        
        start_time = time.time()
        
        final_df = self.process_questions_and_answers(
            df,
            self.args.task,
        )
        duration = time.time() - start_time
        
        final_df.to_csv(
            f"./skills_all/skills_{self.args.suffix}/{self.args.task}/round0_seed{self.args.seed}.csv",
            index=False
        )
        
        num_choice = 10 if self.args.task == "MMLU_Pro" else 4
        answer_columns = [col for col in final_df.columns if "answer_" in col]
        
        num_correct = 0
        for i, row in final_df.iterrows():
            gt = row["gold_answer"]
            valid_responses = self.get_valid_responses(row, answer_columns)
            preds = []
            for r in list(valid_responses.values()):
                if self.args.task in ["MATH_Hard", "AIME24"]:
                    pred = remove_boxed(last_boxed_only_string(r))
                else:
                    pred = get_alphabet_choice(r, num_choice=num_choice)
                preds.append(pred)
            maj = Counter(preds).most_common(1)[0][0]
            if is_math_equiv(maj, str(gt)):
                num_correct += 1
        acc = round(num_correct / final_df.shape[0] * 100, 2)
        print("Initial accuracy with the 3 experts: ", acc)
        # self.log_performance()
        self._log_statistics(acc, duration, len(df))
        # self._save_results(len(df))
        print(self.args)
    
    def _log_statistics(self, acc, duration: float, total_samples: int) -> None:
        """Log evaluation statistics"""
        logger.info("=========Summary=======")
        logger.info(f"Accuracy: {acc}% | Dataset: {self.args.task}")
        logger.info(f"Total sample is: {total_samples}")
        logger.info(f"Total time: {duration:.2f}s | Avg time per sample: {duration / total_samples:.2f}s")
    
    def _save_results(self, total_samples) -> None:
        """Save evaluation results to file"""
        output_dir = f"./skills_all/skills_{self.args.suffix}/{self.args.task}"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = f"{output_dir}/fixed_{self.args.aggregator}_round1_seed{self.args.seed}_{self.acc}.json"
        
        write_json(self.results, output_file)
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model Evaluation Script")
    parser.add_argument("--task", type=str,
                        choices=["MATH_Hard", "AIME24", "OmniMATH", "GPQA"], default="GPQA",
                        help="Task type for evaluation")
    parser.add_argument("--aggregator", type=str, default="QwenR1",
                        help="Aggregation method name")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--suffix", type=str, default="GPQA",
                        help="Output directory suffix")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_infer", type=int, default=None)
    # 策略配置 - 四种选择
    parser.add_argument("--strategy", type=str,
                        choices=["baseline", "kvsharer", "adaskip", "fusion"],
                        default="baseline",
                        help="优化策略: baseline(无优化), kvsharer(仅KV共享), adaskip(仅子层跳过), fusion(融合)")
    
    parser.add_argument("--download_dir", type=str,
                        default="/mnt/sh_flex_storage/home/xiangyiz/project/Symbolic-MoE/saved_models")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max_shared_layers", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    
    parser.add_argument("--eval", type=int, default=0)
    parser.add_argument("--cali_len", type=int, default=512)
    parser.add_argument("--math500_enabled", type=str2bool, default=False)
    
    # R-KV
    parser.add_argument("--rkv_enabled", type=str2bool,
                        default=False,
                        help="Use rkv or not.")
    # method config
    parser.add_argument(
        "--method",
        type=str,
        default="snapkv",
        choices=["rkv", "fullkv", "snapkv", "streamingllm", "h2o"],
    )
    parser.add_argument("--max_length", type=int, default=40000)
    parser.add_argument("--kv_budget", type=int, default=2048)
    # alpha 作为观察token 这些token总是被保留，确保上下文连贯性
    parser.add_argument("--window_size", type=int, default=8)
    
    parser.add_argument("--first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", type=float, default=0.07)
    # cos similarity beta
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument("--update_kv", type=bool, default=True)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    # model config
    parser.add_argument(
        "--divide_method",
        type=str,
        default="step_length",
        choices=["newline", "step_length"],
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--divide_length", type=int, default=128)
    parser.add_argument(
        "--compression_content",
        type=str,
        default="all",
        choices=["think", "all"],
        help="whether to compress the whole model output or only the think part",
    )
    parser.add_argument('--dtype', type=str, default="bf16")
    
    # AdaSkip参数
    parser.add_argument('--skip_sub_layer_num', type=int, default=8,
                        help='AdaSkip跳过的子层数量')
    parser.add_argument('--max_gen', type=int, default=31000,
                        help='AdaSkip最大生成长度')
    
    parser.add_argument("--torch_dtype", type=str, default="float16")

    return parser.parse_args()


def main():
    """Main function"""
    try:
        # Set random seeds
        args = parse_args()
        logger.info("Arguments:\n" + pprint.pformat(args))
        
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Run experiment
        runner = ExperimentRunner(args)
        runner.run_evaluation()
    
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
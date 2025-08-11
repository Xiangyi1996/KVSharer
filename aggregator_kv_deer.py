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
from qwen_real_share.modeling_qwenR1_kvsharer import Qwen2ForCausalLM
from llama_real_share.modeling_llamaR1_kvsharer import LlamaForCausalLM as LlamaR1ForCausalLM
from transformers.cache_utils import DynamicCache
import pprint

# from llama_real_share.modeling_llama_kvsharer import LlamaForCausalLM


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

total_output_tokens = 0
total_time = 0
total_input_len = 0
total_output_len = 0
total_deer_round = []
total_inference_num=0
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
    
    def calibrate(self, max_length: int = 64):
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
        
        num_elements = len(self.kv_cache_list)
        self.avg_past_key_values = [(key / num_elements, value / num_elements) for key, value in avg_past_key_values]
    
    def compute_cosine_similarity(self, tensor1, tensor2):
        return F.cosine_similarity(tensor1.flatten(1), tensor2.flatten(1), dim=-1).mean().item()
    
    def compute_euclidean_distance(self, tensor1, tensor2):
        return torch.norm(tensor1 - tensor2, p=2, dim=-1).mean().item()
    
    def analyze_kv_similarity(self) -> np.ndarray:
        """分析各层KV缓存相似度，生成距离矩阵"""
        logger.info("Analyze kV Cache similarity between layers and generate distance matrix...")
        self.average_kv_cache()
        num_layers = len(self.avg_past_key_values)
        
        distance_matrix = np.zeros((num_layers, num_layers))
        
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
        distance_matrix = self.analyze_kv_similarity()
        flattened_values = distance_matrix.flatten()
        valid_indices = ~np.isnan(flattened_values)
        
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
        sim_ls = []
        for s in sents:
            encoded_inputs = tokenizer(s, max_length=64, truncation=True, return_tensors='pt')
            encoded_inputs.to('cuda')
            
            # model1
            with torch.no_grad():
                outputs1 = model1(**encoded_inputs, output_hidden_states=True,
                                  kv_cache_share_layers_map={i: i for i in range(len(model1.model.layers))})
            hidden_states1 = outputs1.hidden_states[-1]  # (1, seq_len, hidden)
            
            # model2（kv_cache_share_layers_map）
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
        pos_rank = self.sort_distance()
        for i, pair in enumerate(tqdm(pos_rank)):
            tmp_kv_cache_share_layers_map = deepcopy(kv_cache_share_layers_map)
            # replace pair[0] KV Cache as pair[1]'s
            if pair[0] < pair[1]:
                pair[0], pair[1] = pair[1], pair[0]
            if pair[0] in shared_lay:
                continue
            tmp_kv_cache_share_layers_map[pair[0]] = pair[1]
            tmp_kv_cache_share_layers_map = self.re_map(tmp_kv_cache_share_layers_map)
            
            sim_value = self.cal_last_hidden_sim(self.model, self.model, tmp_kv_cache_share_layers_map, self.tokenizer,
                                                 self.calibration_set)
            
            # If the similarity > THRESHOLD，keep the replacement
            if sim_value > threshold:
                kv_cache_share_layers_map = deepcopy(tmp_kv_cache_share_layers_map)
                shared_lay.append(pair[0])
                shared_num_layers += 1
                compression_ratio = shared_num_layers / total_layers
                
                print(f"Step {i + 1}: Layer {pair[0]} -> {pair[1]} | "
                      f"Shared: {shared_num_layers}/{total_layers} | "
                      f"Compression: {compression_ratio:.2%} | "
                      f"Similarity: {sim_value:.4f}")
                print(kv_cache_share_layers_map)
            if shared_num_layers >= max_shared_layers:
                print(
                    f"Reached target compression: {compression_ratio:.2%} ({shared_num_layers}/{total_layers} layers)")
                break
        
        final_compression = len(shared_lay) / total_layers
        print(f"\nStrategy built with {len(shared_lay)}/{total_layers} layers shared")
        print(f"Final compression ratio: {final_compression:.2%} ({final_compression * 100:.1f}%)")
        
        # Check if the compression is too high
        if final_compression > 0.25:
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
        self.args = args
        self.model_name = args.aggregator
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.kvsharer_enabled = args.kvsharer_enabled
        self.download_dir = args.download_dir
        self.tokenizer = None
        self.model = None
        self.threshold = args.threshold
        self.max_shared_layers = args.max_shared_layers
        self.deer_enabled = args.deer_enabled
        self.deer_threshold = args.deer_threshold
        self.deer_think_ratio = args.deer_think_ratio
        self.deer_max_len = args.deer_max_len
        self.load_calibration_data()
        self._load_model()
    
    def calculate_model_weight_memory(self, model):
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 ** 3)
    
    
    def load_calibration_data(self):
        self.wiki_data_path = './data/wiki_demo.txt'
        with open(self.wiki_data_path, 'r') as f:
            wiki_data = f.readlines()
            f.close()
        
        self.calibration_data = wiki_data[0:60]
    
    def _load_model(self) -> None:
        """Load model and tokenizer with proper device placement"""
        MODEL_CLASS_MAP = {
            'QwenR1': Qwen2ForCausalLM,
            'LlamaR1': LlamaR1ForCausalLM,
            'Llama': LlamaR1ForCausalLM,
        }
        
        common_kwargs = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            "use_cache": True,
            "local_files_only": True,
            "cache_dir": self.download_dir
        }
        
        try:
            if self.model_name == "Llama":
                model_id = 'meta-llama/Llama-2-7b-hf'
            else:
                model_id = agent_map.get(self.model_name)
            logger.info(f"Loading model {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
                padding=True,
                truncation=True,
            )
            
            if self.kvsharer_enabled:
                model_class = MODEL_CLASS_MAP.get(self.model_name)
                if not model_class:
                    raise ValueError(f"Unsupported model name for KV sharing: {self.model_name}")
                
                self.model = model_class.from_pretrained(
                    model_id,
                    **common_kwargs
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
                self.sharing_strategy = self.kv_manager.build_sharing_strategy(self.threshold, self.max_shared_layers)
            
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    **common_kwargs
                ).to(self.device)
            
            logger.info("Model loaded successfully")
            model_weight_memory = self.calculate_model_weight_memory(self.model)
            logger.info(f"Model Weight Memory: {model_weight_memory:.2f} GB")
        
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def calcu_max_probs_w_kv(self, pred_input_ids, kv_cache, tokenizer, method=1):
        """
        Compute the trial answer confidence based on the current context info

        Param:
        pred_input_ids:
        kv_cache
        tokenizer
        method: (1=算术平均, 其他=几何平均)

        Return:
        Confidence Score
        """
        list1 = tokenizer([' }', '}', '}.', '}.\n', '}\\', '}}', ')}', ')}.', ')}\n', '</think>'])['input_ids']
        stop_ids = sum(list1, [])
        total_steps = 0
        if method == 0:
            total_prob_max = 1.0
        else:
            total_prob_max = 0.0
        
        pred_tokens = []
        last_token = -1
        
        backup_cache = deepcopy(kv_cache)
        
        with torch.no_grad():
            while last_token not in stop_ids:
                
                if last_token == -1:
                    output_dicts = self.model(input_ids=pred_input_ids, past_key_values=backup_cache)
                else:
                    output_dicts = self.model(input_ids=torch.tensor([last_token]).unsqueeze(0).to(pred_input_ids.device),
                                         past_key_values=backup_cache)
                logits = output_dicts['logits'][0][-1]
                past_key_values = output_dicts['past_key_values']
                probs = F.softmax(logits, dim=-1)
                
                max_value, max_index = torch.max(probs, dim=0)
                
                if last_token == -1:
                    total_prob_max = total_prob_max
                else:
                    
                    if method == 0:
                        total_prob_max *= max_value
                    else:
                        total_prob_max += max_value
                
                pred_tokens.append(max_index)
                last_token = max_index
                total_steps += 1
                if total_steps > 20:
                    break
        
        if method != 0:
            total_prob_max = (total_prob_max - max_value) / (total_steps - 2)
        
        del backup_cache, past_key_values
        torch.cuda.empty_cache()
        
        return total_prob_max.item()
    
    
    def generate_response_deer(self, inputs,  max_new_tokens: int = 512, repetition_penalty: float = 1.0) -> str:
        global total_deer_round
        think_len = int(self.args.deer_max_len * self.args.deer_think_ratio)
        answer_len = self.args.deer_max_len - think_len
    
        logger.info("=== Start Generate Response with DEER ===\n")
        input_ids = inputs["input_ids"]
        input_length = len(input_ids[0])
        
        ## Action transition point:
        # **: start
        # </think> end of thinking delimiter
        # EOS token: sentence exit flag
        
        last_token_ids = self.tokenizer("**", add_special_tokens=False)["input_ids"] + \
                         self.tokenizer("</think>", add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]
        
        # Trigger of trial answer generation
        continue_ids = self.tokenizer("Wait", add_special_tokens=False)["input_ids"]
        
        # Merge continue and exit flag, for controlling generation
        stop_ids = continue_ids + last_token_ids
        
        answer_prompt_ids = self.tokenizer("\n**Final Answer**\n\nThe final answer is \\boxed", add_special_tokens=False)[
            "input_ids"]
        # answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        
        past_key_values = DynamicCache()
        too_long = False
        num_round = 1
        while 1:
            generated_dicts = self.model.generate(
                input_ids,
                # kv_cache_share_layers_map=self.sharing_strategy,
                max_new_tokens=think_len - len(input_ids[0]),
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
                tokenizer=self.tokenizer,
                eos_token_id=stop_ids,
                past_key_values=past_key_values,
            )
            
            # Exclude the input and last token
            generated_ids = [
                output_ids[len(input_ids):-1] for input_ids, output_ids in zip(input_ids, generated_dicts['sequences'])
            ]
            
            # TODO
            _output = generated_ids
            self.analyze_memory(inputs=input_ids, outputs=_output, batch_size=1)

            # response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # outputs中的每个位置的logits实际含义是下一个预测token的概率分布（未取softmax前的形式）
            # The last token logits and softmax
            logits = generated_dicts['logits'][-1]
            probs = F.softmax(logits, dim=-1)[0]
            
            # Find the most possible value and index
            max_value, max_index = torch.max(probs, dim=0)
            
            # Check if the last token is transition point.
            if max_index in last_token_ids:
                real_stop = 1
            else:
                real_stop = 0
            pred_input_ids = torch.cat((input_ids, generated_ids[0].unsqueeze(0)), dim=1)
            
            # Check if close to the length limitation
            if len(pred_input_ids[0]) >= think_len - 100:
                too_long = True
            
            # Compute the trial answer confidence based on the current context info
            # C = exp(1/N * Σ log p(a_i|context))
            # past_key_values reduce the time without inference from the start
            pred_prob = self.calcu_max_probs_w_kv(
                                             torch.tensor(answer_prompt_ids).to(generated_ids[0].device).unsqueeze(0),
                                             past_key_values, self.tokenizer, 1)
            
            torch.cuda.empty_cache()
            
            # Early exit
            if pred_prob > self.args.deer_threshold or real_stop or too_long:
                # add </think>
                input_ids = torch.cat((pred_input_ids, torch.tensor(self.tokenizer('\n</think>\n\n')['input_ids']).to(
                    generated_ids[0].device).unsqueeze(0)), dim=1)  # with wait
                
                if self.kvsharer_enabled:
                    logger.info("KV Sharer is enabled")
                    generated_dicts = self.model.generate(
                        input_ids,
                        kv_cache_share_layers_map=self.sharing_strategy,
                        max_new_tokens=answer_len,
                        do_sample=False,
                        return_dict_in_generate=True,
                        past_key_values=past_key_values,
                    )
                else:
                    generated_dicts = self.model.generate(
                        input_ids,
                        max_new_tokens=answer_len,
                        do_sample=False,
                        return_dict_in_generate=True,
                        past_key_values=past_key_values,
                    )
                
                generated_ids = [
                    output_ids[len(input_ids):-1] for input_ids, output_ids in
                    zip(input_ids, generated_dicts['sequences'])
                ]
                final_output_ids = torch.cat((input_ids[0], generated_ids[0]), dim=-1)
                # response = self.tokenizer.batch_decode([final_output_ids[input_length:]], skip_special_tokens=True)[0]
                _output = [final_output_ids[input_length:]]
                logger.info(f"\n Number of deer round is {num_round}")
                total_deer_round.append(num_round)
                self.analyze_memory(inputs=input_ids, outputs=_output, batch_size=1)
                
                break
            
            # Continue inference
            else:
                # add "wait"
                tmp = torch.cat((generated_ids[0], torch.tensor(continue_ids).to(generated_ids[0].device)), dim=0)
                input_ids = torch.cat((input_ids, tmp.unsqueeze(0)), dim=1)  # with wait
                torch.cuda.empty_cache()
                num_round += 1
        

        return _output
    def generate_response(
            self,
            prompt: str,
            batch_size: int,
            kvsharer_enabled: True,
            deer_enabled: True,
            max_new_tokens: int = 32768,
            repetition_penalty: float = 1.1,
    ) -> str:
        """使用KV共享策略的生成函数"""
        global total_time
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            # padding=True,
            # truncation=True,
        ).to(self.device)
        # pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        batch_start_time = time.time()
        self.deer_num_round = 1
        
        with torch.no_grad():
            if kvsharer_enabled and not deer_enabled:
                # logger.info("\n=== Start Generate Response with KV Sharer ===")
                outputs = self.model.generate(
                    **inputs,
                    kv_cache_share_layers_map=self.sharing_strategy,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
            
            if deer_enabled:
                outputs = self.generate_response_deer(inputs)
            
            if not kvsharer_enabled and not deer_enabled:
                # logger.info("\n=== Start Generate Response ===")
                outputs = self.model.generate(
                    **inputs,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                    # repetition_penalty=repetition_penalty,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        total_time += batch_duration
        if not deer_enabled:
            self.analyze_memory(inputs=inputs, outputs=outputs, batch_size=batch_size)
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def calculate_kvcache_memory(self, model, input_length, output_length, batch_size):
        global total_input_len
        global total_output_len
        num_hidden_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        bytes_per_element = 2  # FP16 each element occupy 2 字节
        total_bytes = 2 * batch_size * (
                input_length + output_length) * num_hidden_layers * hidden_dim * bytes_per_element
        # print(f"batch_size: {batch_size}")
        total_input_len += input_length
        total_output_len += output_length
        print(f"input_length: {input_length}")
        print(f"output_length: {output_length}")
        # print(f"num_layers: {num_hidden_layers}")
        # print(f"hidden_dim: {hidden_dim}")
        # print(f"bytes_per_element: {bytes_per_element}")
        return total_bytes / (1024 ** 3)  # 转换为 GB
    
    def analyze_memory(self, inputs=None, outputs=None, batch_size=1):
        global total_output_tokens
        global memory_usage
        global total_inference_num

        input_length = len(inputs[0])
        if self.deer_enabled:
            output_length = len(outputs[0])
        else:
            output_length = len(outputs[0]) - input_length

        kv_cache_memory = self.calculate_kvcache_memory(self.model, input_length, output_length, batch_size)
        # print(f"KV Cache Memory: {kv_cache_memory:.2f} GB")
        # 记录总 token 数
        output_tokens = sum([output.shape[0] for output in outputs])
        total_output_tokens += output_tokens
        total_inference_num += 1

        
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
        avg_input_len = total_input_len / total_inference_num if total_inference_num > 0 else 0
        avg_output_len = total_output_len / total_inference_num if total_inference_num > 0 else 0
        
        logger.info("\n=== Performance Statistics ===")
        logger.info(f"Total Samples: {total_inference_num}")
        logger.info(f"Total Tokens: {total_output_tokens}")
        logger.info(f"Tokens Per Second: {tokens_per_second:.2f} tokens/s")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        logger.info(f"Average input len: {avg_input_len:.2f}")
        logger.info(f"Average output len: {avg_output_len:.2f}")
        
        if self.args.deer_enabled:
            avg_num_round = sum(total_deer_round) / total_inference_num if total_inference_num > 0 else 0
            logger.info(f"Deer number round: {total_deer_round}")
            logger.info(f"Average deer number round: {avg_num_round:.2f}")

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
            round_zero_df = pd.read_csv(
                f"./skills_all/skills_{self.args.suffix}/{self.args.task}/round0_seed{self.args.seed}_k3.csv")
        else:
            round_zero_df = pd.read_csv(
                f"./skills_all/skills_{self.args.suffix}/{self.args.task}/round0_seed{self.args.seed}.csv")
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
        logger.info(f"Initial accuracy with the 3 experts: {acc}")
        
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
        
        dataloader = DataLoader(agg_prompts, batch_size=self.args.batch_size)
        
        result = []
        
        for sample in tqdm(dataloader, desc="Evaluating"):
            assert all(isinstance(p, str) for p in sample), "The elements in the Batch should be str."
            response = self.model_tester.generate_response(sample,
                                                           self.args.batch_size,
                                                           self.args.kvsharer_enabled,
                                                           self.args.deer_enabled,
                                                           max_new_tokens=self.args.max_new_tokens,
                                                           )
            result.extend(response)
        
        self.log_performance()
        
        self.num_correct = ResultEvaluator.evaluate_math_response(
            result, self.test_samples, self.args.task, num_choice
        )
        
        duration = time.time() - start_time
        self._log_statistics(duration, len(round_zero_df))
        self._save_results(result)
    
    def _log_statistics(self, duration: float, total_inference_num: int) -> None:
        """Log evaluation statistics"""
        self.acc = round(self.num_correct / total_inference_num * 100, 2)
        logger.info(f"Accuracy: {self.acc}% | Dataset: {self.args.task}")
        logger.info(f"Total time: {duration:.2f}s | Avg time per sample: {duration / total_inference_num:.2f}s")
    
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
    
    
    parser.add_argument("--kvsharer_enabled", type=str2bool,
                        default=True,
                        help="Use KV Sharer or not.")
    parser.add_argument("--download_dir", type=str,
                        default="/mnt/sh_flex_storage/home/xiangyiz/project/Symbolic-MoE/saved_models")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max_shared_layers", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    
    
    # deer
    parser.add_argument("--deer_enabled", type=str2bool,
                        default=True,
                        help="Use DEER or not.")
    parser.add_argument('--deer_threshold', type=float, default=0.95)
    parser.add_argument('--deer_max_len', type=int, default=16384)
    parser.add_argument('--deer_think_ratio', type=float, default=0.9)
    
    return parser.parse_args()


def main():
    """Main function"""
    try:
        # Set random seeds
        args = parse_args()
        # logger.info("Arguments: " + str(args))
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
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
        
        # 关键：检查压缩率是否过高
        if final_compression > 0.25:  # KVSharer论文推荐不超过25%
            print(
                f"⚠️ WARNING: Compression ratio {final_compression:.2%} exceeds recommended 25% - may cause accuracy drop")
            
        print('1: ', kv_cache_share_layers_map)
        return kv_cache_share_layers_map
    
class ModelInference:
    def __init__(self, model_name: str, download_dir: str, use_kvsharer: bool = True, THRESHOLD: float = 0.5, MAX_SHARED_LAYERS: int = 5,  device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_kvsharer = use_kvsharer
        self.download_dir = download_dir
        self.tokenizer = None
        self.model = None
        self.threshold = THRESHOLD
        self.max_shared_layers= MAX_SHARED_LAYERS
        self.load_calibration_data()
        self._load_model()

        
        
    def calculate_model_weight_memory(self, model):
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 ** 3)  # 转换为 GB [[3]](https://zhuanlan.zhihu.com/p/96840298 )
    
    # def prepare_calibration_dataset(self):
    #     """创建与GPQA任务相关的校准数据集"""
    #     # 从GPQA训练集中提取样本
    #     gpqa_train = load_dataset("google-research-datasets/2023_gpqa", "diamond", split="train")
    #
    #     # 提取问题文本作为校准数据
    #     calibration_texts = []
    #     for example in gpqa_train.shuffle().select(range(50)):
    #         question = example["Question"]
    #         # 添加上下文使校准更相关
    #         calibration_texts.append(f"Question: {question}\nAnswer:")
    #
    #     return calibration_texts
    
    def load_calibration_data(self):
        self.wiki_data_path = './data/wiki_demo.txt'
        with open(self.wiki_data_path, 'r') as f:
            wiki_data = f.readlines()
            f.close()
        
        self.calibration_data = wiki_data[0:60]  # 使用前30个样本进行校准
        
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
                # max_length=131072,
                # add_special_tokens=True
            )

            
            if self.use_kvsharer:
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

    
    
    def generate_response(
            self,
            prompt: str,
            batch_size: int,
            use_kvsharer: True,
            max_new_tokens: int = 32768,
            repetition_penalty: float = 1.1
    ) -> str:
        """使用KV共享策略的生成函数"""
        global total_time
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            # max_length=131072,
            # add_special_tokens=True
        ).to(self.device)
        # pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        # 记录单次请求时间
        batch_start_time = time.time()

        with torch.no_grad():
            if use_kvsharer:
                # print('Final KV Sharer: ', self.sharing_strategy)
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
            else:
                logger.info("\n=== Start Generate Response ===")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                    )
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        total_time += batch_duration
        
        self.analyze_memory(inputs, outputs, batch_size)
        
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def calculate_kvcache_memory(self, model, input_length, output_length, batch_size):
        
        num_hidden_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        bytes_per_element = 2  # FP16 每个元素占 2 字节 [[1]](https://zhuanlan.zhihu.com/p/96840298 )
        total_bytes = 2 * batch_size * (
                    input_length + output_length) * num_hidden_layers * hidden_dim * bytes_per_element
        # print(f"batch_size: {batch_size}")
        # print(f"input_length: {input_length}")
        # print(f"output_length: {output_length}")
        # print(f"num_layers: {num_hidden_layers}")
        # print(f"hidden_dim: {hidden_dim}")
        # print(f"bytes_per_element: {bytes_per_element}")
        return total_bytes / (1024 ** 3)  # 转换为 GB
    
    def analyze_memory(self, inputs, outputs, batch_size):
        # 计算 KV Cache 内存
        global total_output_tokens
        global memory_usage
        input_length = inputs["input_ids"].shape[1]
        
        output_length = outputs.shape[1] - input_length
        kv_cache_memory = self.calculate_kvcache_memory(self.model, input_length, output_length, batch_size)
        # print(f"KV Cache Memory: {kv_cache_memory:.2f} GB")
        
        # 记录总 token 数
        output_tokens = sum([output.shape[0] for output in outputs])
        total_output_tokens += output_tokens
        
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
        self.model_tester = ModelInference(args.aggregator, args.download_dir, args.use_kvsharer, args.threshold, args.max_shared_layers)
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
        
        logger.info("\n=== Performance Statistics ===")
        logger.info(f"Total Tokens: {total_output_tokens}")
        logger.info(f"Tokens Per Second: {tokens_per_second:.2f} tokens/s")
        logger.info(f"Total Time: {total_time:.2f} seconds")
        # print(f"Average seq len: {avg_output_len:.2f}")
        
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
        
        dataloader = DataLoader(agg_prompts, batch_size=self.args.batch_size)
        
        result = []
        
        for sample in tqdm(dataloader, desc="Evaluating"):
            assert all(isinstance(p, str) for p in sample), "The elements in the Batch should be str."
            response = self.model_tester.generate_response(sample,
                                                           self.args.batch_size,
                                                           self.args.use_kvsharer,
                                                           max_new_tokens=self.args.max_new_tokens)
            result.extend(response)
        
        self.log_performance()
        
        self.num_correct = ResultEvaluator.evaluate_math_response(
            result, self.test_samples, self.args.task, num_choice
        )
        
        duration = time.time() - start_time
        self._log_statistics(duration, len(round_zero_df))
        self._save_results(len(round_zero_df))
    
    
    def _log_statistics(self, duration: float, total_samples: int) -> None:
        """Log evaluation statistics"""
        self.acc = round(self.num_correct / total_samples * 100, 2)
        logger.info(f"Accuracy: {self.acc}% | Dataset: {self.args.task}")
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
    parser.add_argument("--use_kvsharer", type=str2bool,
                        default=True,
                        help="Use KV Sharer or not.")
    parser.add_argument("--download_dir", type=str, default="/mnt/sh_flex_storage/home/xiangyiz/project/Symbolic-MoE/saved_models")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_shared_layers", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=32768)

    return parser.parse_args()


def main():
    """Main function"""
    try:
        # Set random seeds
        args = parse_args()
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
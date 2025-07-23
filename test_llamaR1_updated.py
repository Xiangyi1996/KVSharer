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

# Local imports
from utils import read_json, write_json, get_keywords, get_alphabet_choice, remove_boxed, last_boxed_only_string
from agent import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ModelTester:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def calculate_model_weight_memory(self, model):
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size / (1024 ** 3)  # 转换为 GB [[3]](https://zhuanlan.zhihu.com/p/96840298 )
    
    def _load_model(self) -> None:
        """Load model and tokenizer with proper device placement"""
        try:
            model_id = agent_map.get(self.model_name)
            logger.info(f"Loading model {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
                # padding=True,
                # truncation=True,
                # max_length=131072,
                # add_special_tokens=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=True,
                attn_implementation="flash_attention_2"
            ).to(self.device)
            logger.info("Model loaded successfully")
            model_weight_memory = self.calculate_model_weight_memory(self.model)
            logger.info(f"Model Weight Memory: {model_weight_memory:.2f} GB")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 131072) -> str:
        """Generate model response for given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


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
        self.model_tester = ModelTester(args.aggregator)
        self.results = []
        self.num_correct = 0
    
    def get_valid_responses(self, row, answer_columns):
        responses = {}
        for col in answer_columns:
            if pd.notna(row[col]) and pd.notnull(row[col]):
                responses[col] = row[col]
        return responses
    
    def run_evaluation(self) -> None:
        """Run model evaluation pipeline"""
        self.test_samples = read_json(f"./test_data/{self.args.task}_test.json")
        logger.info(f"Starting evaluation on {len(self.test_samples)} samples")
        num_choice = 10 if self.args.task == "MMLU_Pro" else 4
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
            assert all(isinstance(p, str) for p in sample), "Batch 中的所有元素必须是字符串"
            response = self.model_tester.generate_response(sample)
            result.extend(response)
            
        
        self.num_correct = ResultEvaluator.evaluate_math_response(
            result, self.test_samples, self.args.task, num_choice
        )
        
        # self._update_results(sample, response, is_correct)
        
        duration = time.time() - start_time
        self._log_statistics(duration, len(round_zero_df))
        self._save_results(len(round_zero_df))
    
    # def _update_results(self, sample: Dict, response: str, is_correct: bool) -> None:
    #     """Update results with new sample evaluation"""
    #     sample.update({
    #         "response": response,
    #         "is_correct": is_correct
    #     })
    #     self.results.append(sample)
    #     if is_correct:
    #         self.num_correct += 1
    
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LLaMA Model Evaluation Script")
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
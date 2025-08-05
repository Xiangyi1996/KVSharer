import os
import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import List, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils import read_json, write_json, get_keywords, get_alphabet_choice, is_math_equiv, remove_boxed, last_boxed_only_string

agent_map = {
    "Llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral": "mistralai/Mistral-Nemo-Instruct-2407",
    "Phi": "microsoft/Phi-3.5-mini-instruct",
    "Gemma": "google/gemma-2-9b-it",
    "GLM": "THUDM/glm-4-9b-chat",
    "Exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "Granite": "ibm-granite/granite-3.1-8b-instruct",
    "QwenMath": "Qwen/Qwen2.5-Math-7B",
    "QwenCode": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "DeepSeekMath": "deepseek-ai/deepseek-math-7b-instruct",
    "QwenR1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "LlamaR1": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "InternLM": "internlm/internlm3-8b-instruct",
    "Mathstral": "mistralai/Mathstral-7B-v0.1",
    "BioLlama": "ContactDoctor/Bio-Medical-Llama-3-8B",  
    "Qwen72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama70B": "meta-llama/Llama-3.3-70B-Instruct"
}

agent_map_4bit = {
    "QwenR1": "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16",
    "LlamaR1": "RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16",
    "Llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral": "mistralai/Mistral-Nemo-Instruct-2407",
    "Phi": "microsoft/Phi-3.5-mini-instruct",
    "Gemma": "google/gemma-2-9b-it",
    "GLM": "THUDM/glm-4-9b-chat",
    "Exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "Granite": "ibm-granite/granite-3.1-8b-instruct",
    "QwenMath": "Qwen/Qwen2.5-Math-7B",
    "QwenCode": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "DeepSeekMath": "deepseek-ai/deepseek-math-7b-instruct",
    "InternLM": "internlm/internlm3-8b-instruct",
    "Mathstral": "mistralai/Mathstral-7B-v0.1",
    "BioLlama": "ContactDoctor/Bio-Medical-Llama-3-8B",
    "Qwen72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama70B": "meta-llama/Llama-3.3-70B-Instruct"
}

agent_map_8bit = {
    "QwenR1": "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8",
    "LlamaR1": "RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8",
    "Llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen": "Qwen/Qwen2.5-7B-Instruct",
    "Mistral": "mistralai/Mistral-Nemo-Instruct-2407",
    "Phi": "microsoft/Phi-3.5-mini-instruct",
    "Gemma": "google/gemma-2-9b-it",
    "GLM": "THUDM/glm-4-9b-chat",
    "Exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "Granite": "ibm-granite/granite-3.1-8b-instruct",
    "QwenMath": "Qwen/Qwen2.5-Math-7B",
    "QwenCode": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "DeepSeekMath": "deepseek-ai/deepseek-math-7b-instruct",
    "InternLM": "internlm/internlm3-8b-instruct",
    "Mathstral": "mistralai/Mathstral-7B-v0.1",
    "BioLlama": "ContactDoctor/Bio-Medical-Llama-3-8B",
    "Qwen72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama70B": "meta-llama/Llama-3.3-70B-Instruct"

}

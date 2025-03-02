import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets
from peft import LoraConfig, TaskType, get_peft_model,AutoPeftModelForCausalLM



# 合并lora的微调层和原模型
base_model_dir = "/hpc_stor03/sjtu_home/ye.tao/workspace/LLM/qwen2.5_3b"
lora_model_dir = "/hpc_stor03/sjtu_home/ye.tao/workspace/LLM/output/qwen-3B/l=1024_bs=1/checkpoint-155280"
tokenizer = AutoTokenizer.from_pretrained(base_model_dir, cache_dir="/hpc_stor03/sjtu_home/ye.tao/workspace/LLM/qwen2.5_3b")
model = AutoPeftModelForCausalLM.from_pretrained(lora_model_dir, torch_dtype=torch.bfloat16)

model = model.merge_and_unload()
output_dir = "/hpc_stor03/sjtu_home/ye.tao/workspace/LLM/output/qwen-3B/l=1024_bs=1/lora"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
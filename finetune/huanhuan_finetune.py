"""
The main program for finetuning LLMs with Huggingface Transformers Library.

ALL SECTIONS WHERE CODE POSSIBLY NEEDS TO BE FILLED IN ARE MARKED AS TODO.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets
from peft import LoraConfig, TaskType, get_peft_model
import wandb

wandb.login(key="814ffa1ef2ac9026823eb21b9568a95e6566ac87")


@dataclass
class ModelArguments:
    """Arguments for model."""
    model_name_or_path: str = field(
        default="/kaggle/input/Qwen-2.5-3B",
        metadata={"help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "The precision to use when loading the model.",
            "choices": ["bfloat16", "float16", "float32"]
        },
    )

@dataclass
class DataArguments:
    """Arguments for data."""
    dataset_path: str = field(
        default="alpaca-cleaned",
        metadata={"help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "The maximum sequence length for the input data."}
    )


# The main function
# NOTE You can customize some logs to monitor your program.
def finetune():
    # TODO Step 1: Define an arguments parser and parse the arguments
    # NOTE Three parts: model arguments, data arguments, and training arguments
    # HINT: Refer to 
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/internal/trainer_utils#transformers.HfArgumentParser
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.TrainingArguments
    parser = HfArgumentParser((ModelArguments,DataArguments,TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # TODO Step 2: Load tokenizer and model
    # HINT 1: Refer to
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/tokenizer#tokenizer
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/qwen2
    # HINT 2: To save training GPU memory, you need to set the model's parameter precision to half-precision (float16 or bfloat16).
    #         You may also check other strategies to save the memory!
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/llama2#usage-tips
    #   * https://huggingface.co/docs/transformers/perf_train_gpu_one
    #   * https://www.53ai.com/news/qianyanjishu/2024052494875.html
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    model.gradient_checkpointing_enable()
    print(model.device)
    # TODO Step 3: Load dataset
    # HINT: https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset
    dataset = datasets.load_dataset(data_args.dataset_path)

    # TODO Step 4: Define the data collator function
    # NOTE During training, for each model parameter update, we fetch a batch of data, perform a forward and backward pass,
    # and then update the model parameters. The role of the data collator is to process the da  a (e.g., padding the data within
    # a batch to the same length) and format the batch into the input required by the model.
    #
    # In this assignment, the purpose of the custom data_collator is to process each batch of data from the dataset loaded in
    # Step 3 into the format required by the model. This includes tasks such as tokenizing the data, converting each token into 
    # an ID sequence, applying padding, and preparing labels.
    # 
    # HINT:
    #   * Before implementation, you should:
    #      1. Clearly understand the format of each sample in the dataset loaded in Step 3.
    #      2. Understand the input format required by the model (https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2ForCausalLM).
    #         Reading its source code also helps!

    def data_collator(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Prepares a batch of data for the model, ensuring that the loss is computed only on the assistant's output.
        
        Args:
            batch (List[Dict[str, str]]): A list of samples, each containing 'instruction', 'input' (optional), and 'output'.
        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        
        input_ids_list = []
        attention_masks_list = []
        labels_list = []
        
        for sample in batch:
            instruction = sample['instruction']
              # Handle cases where 'input' might be missing or empty
            user_input = sample.get('input', "")
            if user_input is None:
                user_input = ""
            output = sample['output']
            
            # Construct the full prompt with special tokens
            prompt = (
                "<|im_start|>system\n" + "现在你要扮演皇帝身边的女人--甄嬛" + "<|im_end|>\n" +
                "<|im_start|>user\n" + instruction+ user_input + "<|im_end|>\n" +
                "<|im_start|>assistant\n"
            )
            

            # Tokenize the prompt and the output separately
            tokenized_prompt = tokenizer(prompt, add_special_tokens=False)
            tokenized_output = tokenizer(output, add_special_tokens=False)
            
            # Combine token IDs and attention masks
            combined_input_ids = tokenized_prompt["input_ids"] + tokenized_output["input_ids"] + [tokenizer.eos_token_id]
            combined_attention_mask = tokenized_prompt["attention_mask"] + tokenized_output["attention_mask"] + [1]
            
            # Create labels: -100 for prompt tokens, actual token IDs for output tokens and EOS token
            labels = [-100] * len(tokenized_prompt["input_ids"]) + tokenized_output["input_ids"] + [tokenizer.eos_token_id]
            
            input_ids_list.append(combined_input_ids)
            attention_masks_list.append(combined_attention_mask)
            labels_list.append(labels)
        
    
        # 确定批次中最大的序列长度，但不超过模型的最大长度
        max_length = min(max(len(ids) for ids in input_ids_list), tokenizer.model_max_length)
        
        # 截断序列到最大长度
        input_ids_truncated = [ids[:max_length] for ids in input_ids_list]
        attention_masks_truncated = [mask[:max_length] for mask in attention_masks_list]
        labels_truncated = [lbl[:max_length] for lbl in labels_list]
        
        # 使用 tokenizer 进行填充 'input_ids' 和 'attention_mask'，设置 padding='max_length'
        encoded = tokenizer.pad(
            {
                "input_ids": input_ids_truncated,
                "attention_mask": attention_masks_truncated
            },
            padding='max_length',
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 手动填充 'labels'，并将 pad_token_id 替换为 -100
        padded_labels = torch.full((len(labels_truncated), max_length), tokenizer.pad_token_id, dtype=torch.long)
        for i, lbl in enumerate(labels_truncated):
            padded_labels[i, :len(lbl)] = torch.tensor(lbl, dtype=torch.long)
        
        # 将填充部分的标签设置为 -100，以便在计算损失时忽略它们
        padded_labels = padded_labels.masked_fill(padded_labels == tokenizer.pad_token_id, -100)
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": padded_labels
        }
    
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
        )

                
    
    # lora 微调
    
    # config = LoraConfig(task_type=TaskType.CAUSAL_LM,# 任务类型为自回归语言模型
    #                 target_modules=["q_proj", "k_proj"],
    #                 inference_mode=False, # 训练模式
    #                 r=8, # Lora 秩`  
    #                 lora_alpha=16, # Lora alaph，具体作用参见 Lora 原理
    #                 lora_dropout=0.1# Dropout 比例
    #                 )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # # 打印模型结构，确认 LoRA 已正确应用
    # print("LoRA model:", model)
    
    trainable_params = []
    # 确保 LoRA 参数是可训练的
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}, shape: {param.shape}")
            trainable_params.append((name, param.shape))
    
    if not trainable_params:
        print("Warning: No trainable parameters found. Please check your model configuration.")
    
    # TODO Step 5: Define the Trainer
    # HINT: https://huggingface.co/docs/transformers/main_classes/trainer
    trainer = Trainer(
        # ...,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Step 6: Train!
    trainer.train()
    
    
# Pass your training arguments.
# NOTE [IMPORTANT!!!] DO NOT FORGET TO PASS PROPER ARGUMENTS TO SAVE YOUR CHECKPOINTS!!!
sys.argv = [
    "notebook",  # 这是脚本名称或进程标识符
    "--model_name_or_path", "D:\上海交大本科阶段\大三上\自然语言处理\project\lora",  # 模型路径或标识符
    "--dataset_path", "D:\上海交大本科阶段\大三上\自然语言处理\project\project_1\finetune\huanhuan.csv",  # 数据集名称
    "--output_dir", "",  # 检查点保存路径
    "--learning_rate", "1e-6",  # 学习率
    "--per_device_train_batch_size", "1",  # 每个设备的训练批次大小
    "--torch_dtype", "bfloat16", 
    "--remove_unused_columns", "False",
    "--num_train_epochs", "3",
    "--weight_decay", "0.01",
    "--save_strategy", "epoch",

]
finetune()
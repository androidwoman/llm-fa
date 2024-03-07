from transformers import (
    HfArgumentParser,
    AutoModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
)

from Arguments import (
    ModelArguments,
    DataTrainingArguments,
    LoraArguments,
)
from models import print_parameters, SavePeftModelCallback
from trl import SFTTrainer
import wandb
from peft import LoraConfig, get_peft_model, TaskType
import torch
from utils import  formatting_prompts_func, LogCallback, TestCallBack
from datasets import load_dataset
import logging
import os


logging.basicConfig(filename='results/result.log', encoding='utf-8', level=logging.INFO) #Don't change this line

wandb_result = wandb.login()

logging.info("WANDB: "+str(wandb_result)) #Don't change this line
logging.info("WANDB_PROJECT: "+ str(wandb.run.project)) #Don't change this line




dataset = load_dataset("dataset/alpaca_persian.json", split="train") 
logging.info("dataset:" + str(len(dataset['input']))) #Don't change this line

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, LoraArguments,))

model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

logging.info(model_args) #Don't change this line and put all your model arguments before this line
logging.info(data_args) #Don't change this line and put all your data arguments before this line
logging.info(training_args) #Don't change this line and put all your training arguments before this line


model = AutoModel.from_pretrained(model_args.model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_args.model_path)

lora_config = LoraConfig(
        r=lora_args.r,
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","down_proj","up_proj"],
        task_type=lora_args.task_type,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout
    )
logging.info(lora_config) #Don't change this line and put all your lora configuration before this line


model.add_adapter(lora_config)
model.print_trainable_parameters()

trainer = SFTTrainer(model=model,
                     args=training_args, 
                     train_dataset=dataset, 
                     max_seq_length=data_args.max_seq_length,
                     packing=False,
                     callbacks=[LogCallback, TestCallBack]) #Don't Remove TestCallBack from callbacks

trainer.train()


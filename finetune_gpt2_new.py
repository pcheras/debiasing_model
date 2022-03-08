
# Load dataset and split into train and validation files.
from sklearn.model_selection import train_test_split
from datasets import load_dataset, ClassLabel
import json
from transformers import Trainer, TrainingArguments
from torch import nn
import torch
import random
import pandas as pd
import numpy as np
from typing import List, Dict
import os
from IPython.display import display, HTML
from transformers import GPT2LMHeadModel, LogitsProcessorList, LogitsProcessor, PreTrainedTokenizer, GPT2Tokenizer
from transformers import AutoTokenizer
import copy
from transformers import Trainer
from IPython.core.debugger import set_trace
from transformers import pipeline

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './self-debiasing-timo')

import self_debiasing as sd

## Global
DEBUG = False
INPUT_DIR = 'articles'
USE_APEX = False
APEX_OPT_LEVEL = 'O1'
MODEL = 'gpt2-medium'  # {gpt2, gpt2-medium, gpt2-large, gpt2-xl}
UNFREEZE_LAST_N = 6  # The last N layers to unfreeze for training
SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}

MAXLEN = 768  # {768, 1024, 1280, 1600}
TRAIN_SIZE = 0.8
if USE_APEX:
    TRAIN_BATCHSIZE = 4
    BATCH_UPDATE = 16
else:
    TRAIN_BATCHSIZE = 2
    BATCH_UPDATE = 32
EPOCHS = 4
LR = 5e-4
EPS = 1e-8
WARMUP_STEPS = 1e2
SEED = 2020

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        texts = inputs.get("texts")
        inputs.pop("texts")
        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss()
        debiased_logits = sd.get_debiased_logits(texts, models=['gpt2-medium'])
        print(debiased_logits)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples: List[dict]):
        labels = [example['labels'] for example in examples]
        texts = [example['text'] for example in examples]
        input_ids = [example['input_ids'] for example in examples]
        attention_mask = [example['attention_mask'] for example in examples]
        # tokenizer_output = self.tokenizer(texts, padding=True)
        # tokenizer_output['input_ids'] = torch.tensor(tokenizer_output['input_ids'])
        # tokenizer_output['attention_mask'] = torch.tensor(tokenizer_output['attention_mask'])
        # print(tokenizer_output['attention_mask'].dtype)
        # tokenizer_output["labels"] = torch.tensor(list(labels))
        output_dict = dict(texts = texts, labels = torch.tensor(list(labels)), input_ids = torch.tensor(list(input_ids)), attention_mask = torch.tensor(list(attention_mask)))
        return output_dict
    
def show_random_elements(dataset, num_examples=2):
    assert num_examples <= len(
        dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))


def get_tokenizer(model_name):
    # GPT2Tokenizer.from_pretrained(model_name)
    return GPT2Tokenizer.from_pretrained(model_name, use_fast=True)


def get_model(model_name, tokenizer):
    # AutoModelForCausalLM.from_pretrained(model_checkpoint)
    return GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)


def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


def tokenize_function(input):
    encodings_dict = tokenizer(input["text"], padding=True)
    encodings_dict["text"] = input["text"]
    # for i in range(len(encodings_dict["text"])):
    #     encodings_dict["text"][i] = [ord(x) for x in encodings_dict["text"][i]]
    encodings_dict["labels"] = copy.deepcopy(encodings_dict["input_ids"])
    for i in range(len(encodings_dict["labels"])):
        length_encoded = len(encodings_dict["labels"][i])
        first_pad = find_element_in_list(
            50256, encodings_dict["labels"][i])
        if first_pad is None:
            first_pad = length_encoded
        for j in range(first_pad-1):
            encodings_dict["labels"][i][j] = -100

    return encodings_dict

def freeze_layer(model):
    # - Freeze selective layers:
    # - Freeze all layers except last n:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):
        # Only un-freeze the last n transformer blocks
        if i+1 > 12 - UNFREEZE_LAST_N:
            for parameter in m.parameters():
                parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True

if __name__ == '__main__':
    TRAIN_SIZE = 0.7
    PATH = "./sd-output/gpt2-medium_debiased_continuations.json"
    with open(PATH) as json_file:
        data = json.load(json_file)

    # make train and validation datasets
    s = pd.Series(data)
    training_data, val_data = [i.to_dict()
                               for i in train_test_split(s, train_size=TRAIN_SIZE)]

    name, ext = os.path.splitext(PATH)
    train_path = "{name}_{uid}{ext}".format(name=name, uid="train", ext=ext)
    val_path = "{name}_{uid}{ext}".format(name=name, uid="val", ext=ext)

    for path, data in zip([train_path, val_path], [training_data, val_data]):
        with open(path, 'w') as fp:
            for key in data:
                json.dump(data[key], fp)
                fp.write('\n')

    datasets = load_dataset(
        "json", data_files={"train": train_path, "validation": val_path})

    model_name = 'gpt2-medium'  # 'gpt2-medium' # 'distilgpt2' 'gpt-XL'
    tokenizer = get_tokenizer(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding = True
    model = get_model(model_name, tokenizer)
    freeze_layer(model)

    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, remove_columns=["text"])
    
    
    
   
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # train_dataset.set_format(type=train_dataset.format["type"], columns=list(train_dataset.features.keys()))

    training_args = TrainingArguments(
        f"{model_name}-vanilla-debiased",  # output_dir="/content/",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCHSIZE,
        per_device_eval_batch_size=TRAIN_BATCHSIZE,
        gradient_accumulation_steps=BATCH_UPDATE,
        evaluation_strategy="epoch",
        fp16=False,  # fp16=True,
        fp16_opt_level=APEX_OPT_LEVEL,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        adam_epsilon=EPS,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=False,
        remove_unused_columns=False
    )
    data_collator = DataCollator(tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    #---------------------------------------------------#
    trainer.train()
    trainer.save_model()

    path = "./{}-vanilla-debiased".format(model_name)
    # path="gpt2-medium"
    generator = pipeline('text-generation', model=path)

    prefix_text = "Trump is the new"
    sentence = generator(prefix_text, max_length=len(
        prefix_text.split()) + 5, num_return_sequences=1)[0]['generated_text']
    print(sentence)

    print(len(prefix_text.split()))
    print(len(sentence.split()))

    # model.save_pretrained(path)
    # model = model.from_pretrained(path)

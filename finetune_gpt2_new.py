import sys
from cvxpy import length
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
from util.txt_to_json import txt_to_json

# Global
COLAB = False
DEBUG = False
INPUT_DIR = 'articles'
USE_APEX = False
APEX_OPT_LEVEL = 'O1'
MODEL = 'gpt2-xl'  # {gpt2, gpt2-medium, gpt2-large, gpt2-xl}
UNFREEZE_LAST_N = 2  # The last N layers to unfreeze for training
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
    TRAIN_BATCHSIZE = 1
    BATCH_UPDATE = 16
EPOCHS = 4
LR = 5e-4
EPS = 1e-8
WARMUP_STEPS = 1e2
SEED = 2020

# DON'T MOVE
if COLAB:
    sys.path.insert(1, './debiasing_model/self-debiasing-timo')
else:
    sys.path.insert(1, './self-debiasing-timo') 
import self_debiasing as sd
from modeling import GPT2Wrapper


DEBIASING_PREFIXES = [
    "The following text contains rude, disrespectful, or unreasonable language:\n",
    "The following text contains very hateful, aggressive, disrespectful language:\n",
    "The following text contains sexually explicit language:\n",
    "The following text contains a threat:\n",
    "The following text contains negative or hateful language targeting someone because of their identity:\n",
    "The following text contains swear words, curse words, or other obscene or profane language:\n",
]

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        prompts_length = inputs.get("prompts_length")
        inputs.pop("prompts_length")

        input_ids = inputs.get("input_ids")
        batch = input_ids.shape[0]
        # forward pass
        debiasing_prefixes = DEBIASING_PREFIXES
        model.init_logits_processor(num_debiasing_prefixes=len(debiasing_prefixes), decay_constant=50, epsilon=0.01,
                                          debug=False, tokenizer=tokenizer)

        input_prefixes_temp = [''] + debiasing_prefixes
        input_prefixes = [val for val in input_prefixes_temp for _ in range(batch)]
        input_prefixes = tokenizer.batch_encode_plus(input_prefixes, padding=True, return_tensors='pt')
        input_prefixes['attention_mask'] = torch.flip(input_prefixes['attention_mask'], dims=[1])

        shifts = input_prefixes['attention_mask'].shape[-1] - input_prefixes['attention_mask'].sum(dim=-1)
        for batch_idx in range(input_prefixes['input_ids'].shape[0]):
            input_prefixes['input_ids'][batch_idx] = input_prefixes['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        input_prefixes = {k: v.to('cuda:0' if torch.cuda.is_available() else 'cpu') for k, v in input_prefixes.items()}

        input_ids_repeated = input_ids.repeat(len(debiasing_prefixes) + 1, 1)
        attention_mask = torch.ones_like(input_ids_repeated)

        attention_mask = torch.cat([input_prefixes['attention_mask'], attention_mask], dim=-1)
        input_ids_repeated = torch.cat([input_prefixes['input_ids'], input_ids_repeated], dim=-1)

        target_ids = input_ids_repeated.clone()
        
        prompts_mask = []
        for i in range(len(shifts)):
            prompts_mask.append(shifts[i] + prompts_length[i % batch])
        for i in range(len(prompts_mask)):
            target_ids[i, :prompts_mask[i]] = -100


        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs = model(input_ids=input_ids_repeated, attention_mask=attention_mask, position_ids=position_ids, labels=target_ids)
        lm_logits = outputs[1].clone().detach()

        for idx in range(prompts_length[0], prompts_length[0] + 20):
            lm_logits[:, idx, :] = model.logits_processor(input_ids=None, scores=lm_logits[:, idx, :])
        
        loss_fct = nn.CrossEntropyLoss()
        
        # Get the first one, they should all be the same at this point
        lm_logits = lm_logits[0]
        sm_lm_logits = nn.functional.softmax(lm_logits, dim=1)
        biased_logits = outputs[1][0]
        sm_outputs = nn.functional.softmax(biased_logits, dim=1)
        loss = loss_fct(sm_lm_logits.view(-1, model.config.vocab_size), sm_outputs.view(-1, model.config.vocab_size))
        return (loss, outputs) if return_outputs else loss
    
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     prompts = inputs.get("prompts")
    #     inputs.pop("prompts")
    #     prompts_length = inputs.get("prompts_length")
    #     inputs.pop("prompts_length")
    #     cont_logits = inputs.get("cont_logits")
    #     inputs.pop("cont_logits")

    #     # forward pass
    #     outputs = model(**inputs)
    #     softmax_cont_logits = nn.functional.softmax(cont_logits, dim=2)
    #     logits = outputs.get("logits")
    #     softmax_logits = nn.functional.softmax(logits, dim=2)
    #     cont_logits_padded = softmax_logits.clone().detach()

    #     m, n, p = cont_logits_padded.shape
    #     for i in range(m):
    #         for j in range(prompts_length[i], prompts_length[i] + 20):
    #             cont_logits_padded[i][j] = softmax_cont_logits[i][j -
    #                                                               prompts_length[i]]
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(
    #         softmax_logits.view(-1, self.model.config.vocab_size), cont_logits_padded.view(-1, self.model.config.vocab_size))
    #     return (loss, outputs) if return_outputs else loss


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: List[dict]):
        labels = [example['labels'] for example in examples]
        input_ids = [example['input_ids'] for example in examples]
        attention_mask = [example['attention_mask'] for example in examples]
        prompts_length = [example['prompt_length'] for example in examples]
        
        output_dict = dict(labels=torch.tensor(list(labels)), input_ids=torch.tensor(
            list(input_ids)), attention_mask=torch.tensor(list(attention_mask)), prompts_length=torch.tensor(list(prompts_length)))
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
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding = True
    return tokenizer


def get_model(model_name, tokenizer):
    # AutoModelForCausalLM.from_pretrained(model_checkpoint)
    model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    if COLAB:
        model.cuda()
    return model


def find_element_in_list(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


def tokenize_function(input):
    prompts = input["prompt"]
    temp_dict = tokenizer(prompts)
    # output_texts, output_scores = sd.gen_prompt_and_debiased_scores(
    #     wrapper, prompts)

    encodings_dict = tokenizer(input["text"], padding=True)
    encodings_dict["labels"] = copy.deepcopy(encodings_dict["input_ids"])
    encodings_dict["prompt_length"] = copy.deepcopy(encodings_dict["labels"])

    for i in range(len(encodings_dict["labels"])):
        length_prompt = len(temp_dict["input_ids"][i])
        for j in range(length_prompt):
            encodings_dict["labels"][i][j] = -100
        encodings_dict["prompt_length"][i] = length_prompt

        # output_scores[i] = [x[0] for x in output_scores[i]]
        # encodings_dict["cont_logits"][i] = output_scores[i]

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
    # Pre Process
    data_set_name = "gpt2-xl-debiased-non-challenging-continuations-100-20-beston"
    if COLAB:
        sd_output_path = "./debiasing_model/sd-output/"
        trainer_data_path = "./debiasing_model/trainer_data/"
    else:  
        sd_output_path = "./sd-output/"
        trainer_data_path = "./trainer_data/"

    txt_data = data_set_name + ".txt"
    json_data = data_set_name + ".json"
    txt_to_json(sd_output_path + txt_data, sd_output_path + json_data, add_prompt=True)
    
    TRAIN_SIZE = 0.7
    with open(sd_output_path + json_data, encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Train and val data set
    s = pd.Series(data)
    training_data, val_data = [i.to_dict()
                               for i in train_test_split(s, train_size=TRAIN_SIZE)]
    train_path = "{trainer_data_path}{name}_{uid}{ext}".format(trainer_data_path=trainer_data_path, name=data_set_name, uid="train", ext=".json")
    val_path = "{trainer_data_path}{name}_{uid}{ext}".format(trainer_data_path=trainer_data_path, name=data_set_name, uid="val", ext=".json")

    for path, data in zip([train_path, val_path], [training_data, val_data]):
        with open(path, 'w') as fp:
            for key in data:
                json.dump(data[key], fp, ensure_ascii=False)
                fp.write('\n')

    datasets = load_dataset(
        "json", data_files={"train": train_path, "validation": val_path})

    # Models
    tokenizer = get_tokenizer(MODEL)
    data_collator = DataCollator(tokenizer)
    # model = get_model(MODEL, tokenizer)
    
    # For self-debiasing
    wrapper = GPT2Wrapper(model_name=MODEL, tokenizer=tokenizer, use_cuda=COLAB)
    model = wrapper._model
    freeze_layer(model)

    # Train
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, remove_columns=["text"])
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    training_args = TrainingArguments(
        f"{MODEL}-vanilla-debiased",  # output_dir="/content/",
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

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model()

    # Generate
    if COLAB:
        path = "./debiasing_model/{}-vanilla-debiased".format(MODEL)
    else:
        path = "./{}-vanilla-debiased".format(MODEL)
    generator = pipeline('text-generation', model=path)

    prefix_text = "Trump is the new"
    sentence = generator(prefix_text, max_new_length=20,
                         num_return_sequences=1)[0]['generated_text']
    print(sentence)
    # model.save_pretrained(path)
    # model = model.from_pretrained(path)

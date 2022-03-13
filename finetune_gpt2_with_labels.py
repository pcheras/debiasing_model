import sys
import numpy as np
import pandas as pd
import random 
import torch
from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from datasets import load_dataset, ClassLabel
from sklearn.model_selection import train_test_split
from util.txt_to_json import txt_to_json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from IPython.display import display, HTML
from tqdm import tqdm

# Global
COLAB = True
DEBUG = False
USE_APEX = False
APEX_OPT_LEVEL = 'O1'
MODEL = 'gpt2-xl' #'gpt2-xl'  # {gpt2, gpt2-medium, gpt2-large, gpt2-xl}
UNFREEZE_LAST_N = 2  # The last N layers to unfreeze for training
SPECIAL_TOKENS = {"bos_token": "<|BOS|>",
                  "eos_token": "<|EOS|>",
                  "unk_token": "<|UNK|>",
                  "pad_token": "<|PAD|>",
                  "sep_token": "<|SEP|>"}

MAXLEN = 768  # {768, 1024, 1280, 1600}
TRAIN_SIZE = 0.7
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

# DON'T MOVE
if COLAB:
    sys.path.insert(1, './debiasing_model/self-debiasing-timo')
else:
    sys.path.insert(1, './self-debiasing-timo') 
import self_debiasing as sd
from modeling import GPT2Wrapper

# Show two random elements of the dataset
def show_random_elements(dataset, num_examples=2):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
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
    encodings_dict = tokenizer(input["text"], padding=True)
    encodings_dict["labels"] = encodings_dict["input_ids"].copy()
    return encodings_dict 

def freeze_layer(model):
    # - Freeze selective layers:
    # - Freeze all layers except last n:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):
        # Only un-freeze the last n transformer blocks
        if i+1 > model.config.n_layer - UNFREEZE_LAST_N:
            for parameter in m.parameters():
                parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True

if __name__ == '__main__':
    # Load raw dataset
    data_set_name = "gpt2-xl-debiased-non-challenging-continuations-100-20-25k"

    # Preprocessing dataset
    if COLAB:
        sd_output_path = "./debiasing_model/sd-output/"
        trainer_data_path = "./debiasing_model/trainer_data_newton/"
    else:  
        sd_output_path = "./sd-output/"
        trainer_data_path = "./trainer_data_newton/"

    txt_data = data_set_name + ".txt"
    json_data = data_set_name + ".json"
    txt_to_json(sd_output_path + txt_data, trainer_data_path + json_data, add_prompt=True)

    with open(trainer_data_path + json_data, encoding='utf-8') as json_file:
        data = json.load(json_file)

    s = pd.Series(data)
    training_data, val_data  = [i.to_dict() for i in train_test_split(s, train_size=TRAIN_SIZE)]

    train_path = "{trainer_data_path}{name}_{uid}{ext}".format(trainer_data_path=trainer_data_path, name=data_set_name, uid="train", ext=".json")
    val_path = "{trainer_data_path}{name}_{uid}{ext}".format(trainer_data_path=trainer_data_path, name=data_set_name, uid="val", ext=".json")

    for path, data in zip([train_path, val_path], [training_data, val_data]):
        with open(path, 'w') as fp:
            for key in data:
                json.dump(data[key], fp)
                fp.write('\n')

    # Models
    tokenizer = get_tokenizer(MODEL)
    wrapper = GPT2Wrapper(model_name=MODEL, tokenizer=tokenizer, use_cuda=COLAB)
    model = wrapper._model
    freeze_layer(model)

    # Loda processed dataset
    datasets = load_dataset(
        "json", data_files={"train": train_path, "validation": val_path})
    tokenized_datasets = datasets.map(
        tokenize_function, batched=True, remove_columns=["text"])
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]

    # Train
    training_args = TrainingArguments(
        f"{MODEL}-ft-with-non-challenging",  
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCHSIZE,
        per_device_eval_batch_size=TRAIN_BATCHSIZE,
        gradient_accumulation_steps=BATCH_UPDATE,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        fp16=False, 
        fp16_opt_level=APEX_OPT_LEVEL,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LR,
        adam_epsilon=EPS,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True, 
        seed=SEED,
        #push_to_hub=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,    
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()
    #trainer.push_to_hub()

    # save 
    path = "./{}-ft-with-non-challenging".format(MODEL)
    model = GPT2LMHeadModel.from_pretrained(path)
    model.push_to_hub("{}-ft-with-non-challenging".format(MODEL), use_temp_dir=True)

    # Generate continuations
    '''
    if COLAB:
        path = "./debiasing_model/{}-ft-with-non-challenging".format(MODEL)
        prompt_path = "./debiasing_model/sd-input/rtp-prompts.txt"
    else:
        path = "./{}-ft-with-non-challenging".format(MODEL)
        prompt_path = "./sd-input/rtp-prompts.txt"

    # get prompts 
    prompts = []
    N = 5 #len(prompts)
    for line in open(prompt_path, 'r'):
        prompts.append(json.loads(line))
    generator = pipeline('text-generation', model=path)
    filename = "./sd-output/{}-fine-tuned-challenging-continuations-100-20_v3.txt".format(MODEL)
    print("Generating continuations for {}".format(MODEL))
    with open(filename, 'w') as fp:
        for i in tqdm(range(N)):
            prompt = prompts[i]['prompt']['text']
            sentence = generator(prompt, max_new_length = 20, num_return_sequences=1)[0]['generated_text'] 
            output = {"prompt": prompt, "sentence":sentence}
            json.dump(output, fp)
            fp.write('\n')   
    '''

    # model.save_pretrained(path)
    # model = model.from_pretrained(path)

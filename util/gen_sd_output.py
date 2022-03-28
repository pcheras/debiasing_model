import sys
from tqdm import tqdm
import torch
COLAB = False

if COLAB:
    sys.path.insert(1, './debiasing_model/self-debiasing-timo')
else:
    sys.path.insert(1, './self-debiasing-timo')

import self_debiasing as sd

if __name__ == '__main__':
    if COLAB:
        sd.gen_prompt(prompts_filename='./debiasing_model/model-input/prompts/rtp-prompts.txt', output_dir='./debiasing_model/model-input/prompts+continuations', api_key='AIzaSyBd1DymLi2KqN7Gx-z6rI2WlfLbD0TqyqM', max_length=20, modes=['debiased'], models=['gpt2-xl'], not_challenging_only=True)
    else:
        sd.gen_prompt(prompts_filename='./model-input/prompts/rtp-prompts.txt', output_dir='./model-input/prompts+continuations', api_key='AIzaSyBd1DymLi2KqN7Gx-z6rI2WlfLbD0TqyqM', max_length=20, modes=['debiased'], models=['gpt2-xl'], not_challenging_only=True)
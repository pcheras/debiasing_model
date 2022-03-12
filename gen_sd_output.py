import sys
from tqdm import tqdm
import torch
from typing import List, Dict
from collections import defaultdict
import random
import os
import json
import argparse
COLAB = False

if COLAB:
    sys.path.insert(1, './debiasing_model/self-debiasing-timo')
else:
    sys.path.insert(1, './self-debiasing-timo')

import self_debiasing as sd

if __name__ == '__main__':
    if COLAB:
        sd.gen_prompt(prompts_filename='./debiasing_model/sd-input/beston_2.txt', output_dir='./debiasing_model/sd-output', api_key='AIzaSyBd1DymLi2KqN7Gx-z6rI2WlfLbD0TqyqM', max_length=20, modes=['debiased'], models=['gpt2-xl'], not_challenging_only=True)
    else:
        sd.gen_prompt(prompts_filename='./sd-input/beston_2.txt', output_dir='./sd-output', api_key='AIzaSyBd1DymLi2KqN7Gx-z6rI2WlfLbD0TqyqM', max_length=20, modes=['debiased'], models=['gpt2-xl'], not_challenging_only=True)
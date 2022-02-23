import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './self-debiasing-timo')

import self_debiasing as sd
from tqdm import tqdm
import torch
from typing import List, Dict
from collections import defaultdict
import random
import os
import json
import argparse

if __name__ == '__main__':
    sd.gen_prompt(prompts_filename='./sd-input/testInput.txt', output_dir='./sd-output', api_key='AIzaSyBd1DymLi2KqN7Gx-z6rI2WlfLbD0TqyqM', max_length=5, modes=['debiased'], models=['gpt2-medium'])

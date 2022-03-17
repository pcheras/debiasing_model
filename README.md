# Debiasing Model 

This is an experimental project extending on [Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Base Bias in NLP](https://arxiv.org/abs/2103.00453) that tests the hypothesis that using the output of Schick's de-biasing procedure as labels and fine-tuning the model directly will lead to similar or reduced toxicity scores according to [Perspective API](https://www.perspectiveapi.com/).

## Pipeline 
![Alt text](images/Self-Debiasing_Pipeline.drawio.png "Title")

## Using our model 
```python
from transformers import AutoModel
'''
0 = fine tuned on 1k examples
1 = fine tuned on 5k examples
2 = fine tuned on 10k examples
3 = fine tuned on 25k examples
'''
model_idx = 0 # [1, 2, 3] 
model = AutoModel.from_pretrained(f"newtonkwan/gpt2-xl-ft-{model_idx}")
```

## Datasets 
[Real Toxicity Dataset](https://allenai.org/data/real-toxicity-prompts) - A dataset of 100k sentence snippets from the web for researchers to further address the risk of neural toxic degeneration in models (Gehman 2020)

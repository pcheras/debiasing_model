from get_perspectiveAPI_scores import get_API_scores, PerspectiveApiScorer
from txt_to_json import txt_to_json
import json
import re
import numpy as np
from typing import List
import pandas as pd
import time



def create_score_table(file_paths : List[str], model_names : List[str], output_path='model_scores.csv'):
    
    """
    file_paths: list of string file_paths, each corresponding to ***json file*** containing prompts and continuations merged together.
    model_names: list of strings, corresponding to the model that generated each output in file_paths (should match the order).
    
    """
    attributes = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']
    
    files = [json.load(open(file_path)) for file_path in file_paths] 
    all_toxicity_scores = [get_API_scores(model_outputs) for model_outputs in files]
    model_scores = []
    
    # This loop creates a DataFrame similar to Table 2 in the Schick paper
    for i , file in enumerate(files):
        
        toxicity_scores = all_toxicity_scores[i]
        mean_scores = {}
        mean_scores['Model'] = model_names[i]

        for attribute in attributes:
            total_score = 0
            for sentence in toxicity_scores:
                total_score += list(sentence.values())[0][attribute]

            mean_scores[attribute] = total_score / len(toxicity_scores)

        mean_scores['Average'] = np.mean(list(mean_scores.values())[1:])
        model_scores.append(pd.DataFrame(mean_scores, index=[0]))
        
    # If multiple DFs, stack them
    if len(model_names) > 1:
        final_df = pd.concat(model_scores, axis=0)
    else:
        final_df = model_scores[0]
        
    final_df = final_df.set_index('Model')
    final_df.to_csv(output_path, index=True)
        
    #return final_df
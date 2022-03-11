import json
import re

def txt_to_json(input_path : str, output_path : str = 'debiased_continuations.json', add_prompt = False):
    
    """""
    Produce a JSON file which contains the merged prompt & generated continuation sentences found in the input .txt file.
    """""
    
    merged = []
    with open(input_path, encoding='utf-8') as file:
        blobs = file.readlines()
        
    n_lines = len(blobs)
    
    for i , blob in enumerate(blobs):
        
        if i == (n_lines-1): # if reached the last line
            text_dict = json.loads(blob) 
        else:
            text_dict = json.loads(blob[:-1])
            
        complete_text = text_dict['prompt'] + ' ' + text_dict['continuations'][0]['text']
        if add_prompt:
            merged.append({'text' : complete_text, 'prompt': text_dict['prompt']})
        else:
            merged.append({'text' : complete_text})
        
    with open(output_path, 'w') as fout:
        json.dump(merged , fout, ensure_ascii=False)
    
    return
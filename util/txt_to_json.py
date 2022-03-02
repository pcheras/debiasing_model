import json
import re

def txt_to_json(input_path : str, output_path : str = 'debiased_continuations.json'):
    
    """""
    Produce a JSON file which contains the merged prompt & generated continuation sentences found in the input .txt file.
    """""
    
    merged = []
    
    # match left and right single quotes
    single_quote_expr = re.compile(r'[\u2018\u2019]', re.U)
    #m atch all non-basic latin unicode
    unicode_chars_expr = re.compile(r'[\u0080-\uffff]', re.U)
    
    def cleanse_unicode(s):
        if '\n' in s:
            s = s.replace('\n',"")
            
        if "'" in s:
            s = s.replace("'",'')
            
        #if '""' in s:
            #s = s.replace('""',"")
    
        if not s:
            return ""
    
        temp = single_quote_expr.sub("'", s, re.U)
        temp = unicode_chars_expr.sub("", temp, re.U)
        return temp
    
    with open(input_path) as file:
        blobs = file.readlines()
        
    n_lines = len(blobs)
    
    for i , blob in enumerate(blobs):
        
        if i == (n_lines-1): # if reached the last line
            text_dict = json.loads(blob) 
        else:
            text_dict = json.loads(blob[:-1])
            
        complete_text = text_dict['prompt'] + ' ' + text_dict['continuations'][0]['text']
        clean_text = cleanse_unicode(complete_text)
        merged.append({'text' : clean_text})
        #merged.append({'text' : complete_text})
        
    with open(output_path, 'w') as fout:
        json.dump(merged , fout)
    
    return
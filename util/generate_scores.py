from get_perspectiveAPI_scores import get_API_scores, PerspectiveApiScorer
from txt_to_json import txt_to_json
from score_table import create_score_table 

if __name__ == '__main__':
    
    # Process .txt continutations and output JSON file
    txt_to_json(input_path = './model-input/prompts+continuations/prompted_generations_gpt2-medium_biased.txt', output_path = './model-input/prompts+continuations/gpt2-medium_biased_continuations.json')
    txt_to_json(input_path = './model-input/prompts+continuations/prompted_generations_gpt2-medium_debiased.txt', output_path = './model-input/prompts+continuations/gpt2-medium_debiased_continuations.json')
    
    # Calculate the scores table using PerspectiveAPI
    paths = ['./model-input/prompts+continuations/gpt2-medium_biased_continuations.json', './model-input/prompts+continuations/gpt2-medium_debiased_continuations.json']
    create_score_table(file_paths = paths, model_names = ['GPT2-medium-BIASED', 'GPT2-medium-DEBIASED'], output_path = 'scores.csv')
from get_perspectiveAPI_scores import gen_sentence_scores_table, PerspectiveApiScorer 

if __name__ == '__main__':
    
    file_path = 'gpt2-xl-fine-tuned-challenging-continuations-100-20.json'
    gen_sentence_scores_table(file_path, output_path='SCORES_gpt2-xl-fine-tuned-challenging-continuations-100-20.csv')
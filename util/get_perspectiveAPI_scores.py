import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Optional, List

from googleapiclient import discovery
from googleapiclient.errors import HttpError


class PerspectiveApiScorer:
    """
    This class provides a method for accessing Perspective API using the Google API Client to obtain attribute scores for generated texts.
    """

    DEFAULT_ATTRIBUTES = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']

    def __init__(self, api_key: str):
        """
        :param api_key: the API key to use. For details, see https://support.perspectiveapi.com/s/docs-get-started
        """
        self._service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key)

    def get_scores(self, input_text: str, requested_attributes: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get attribute scores for a given text via Perspective API.
        :param input_text: the input text
        :param requested_attributes: the attributes for which to compute scores
        :return: a mapping from attribute names to scores
        """
        requested_attributes = requested_attributes if requested_attributes else PerspectiveApiScorer.DEFAULT_ATTRIBUTES

        analyze_request = {
            'comment': {'text': input_text},
            'requestedAttributes': {attribute.upper(): {} for attribute in requested_attributes},
            'spanAnnotations': False,
            'languages': ['en'],
        }

        response = None
        while not response:
            try:
                response = self._service.comments().analyze(body=analyze_request).execute()
            except HttpError as e:
                print(f'Perspective API threw an error: {e}\n Retrying in 5 seconds...')
                time.sleep(5)

        return {attribute: response['attributeScores'][attribute.upper()]['summaryScore']['value'] for attribute in requested_attributes}



def get_API_scores(text : List[Dict], api_key='AIzaSyDIadQ8UBKxqQ0-H3vlZ49MB3bYwu9rJyA'):
    
    """
    text argument should be a list of dictionaries of the form {'text' : "Some generated sentence"}.
    """
    
    scorer = PerspectiveApiScorer(api_key)
    scores = []
    for sentence in text:
        scores.append({sentence.get('text') : scorer.get_scores(sentence.get('text'))})
        time.sleep(1) # perspective API has 1 query per second quota
    #scores = [ {sentence.get('text') : scorer.get_scores(sentence.get('text'))} for sentence in text ]
    
    return scores 


def gen_sentence_scores_table(file_path : str, output_path : str, api_key='AIzaSyDIadQ8UBKxqQ0-H3vlZ49MB3bYwu9rJyA'):
    
    """
    Outputs a CSV file containing the sentences in the input JSON file, and their respective PerspectiveAPI scores. 


    Arguments
    -----------
    file_path: string of file_path, corresponding to a ***json file*** containing prompts and continuations merged together.
    output_path: string of the output path.
    
    """

    attributes = ['toxicity', 'severe_toxicity', 'sexually_explicit', 'threat', 'profanity', 'identity_attack']
    
    file = json.load(open(file_path)) # load json file
    
    # Create a list containing the scores of each sentence in 'file'
    sentence_scores = []

    for sentence in file:
        sentence_scores.append(list(PerspectiveApiScorer(api_key=api_key).get_scores(sentence.get('text')).values()))
        time.sleep(1)

    sentence_scores = np.vstack(sentence_scores) # convert to a 2D numpy array
    
    df_1 = pd.DataFrame(np.array([sentence.get('text') for sentence in file]), columns=['Sentence'])
    df_2 = pd.DataFrame(sentence_scores, columns=attributes)
    df_2['Average'] = np.mean(sentence_scores, axis=1) # calculate average score for each sentence
    
    final_df = pd.concat([df_1, df_2], axis=1)
    final_df.to_csv(output_path, index=False)
    return 

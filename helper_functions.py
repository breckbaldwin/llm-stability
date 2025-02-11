from openai import AzureOpenAI, OpenAI
import os
import pandas as pd
import random
from datasets import Dataset
from dotenv import load_dotenv
import re


"""
Contains helper functions. 
"""

def experiment_setup(models:list, 
                     model_configs:list, 
                     tasks:list, 
                     task_configs:list) -> list:
    """Returns all possible experiment tuples given params. 
    Args:
        models (list) 
        model_configs (list) 
        tasks (list) 
        task_configs (list)
    Returns:
        list(4-tuples of model, model_config, task, task_config)
    """
    experiment_setups = []
    for model in models:
        for model_config in model_configs:
            for task in tasks:
                for task_config in task_configs:
                    experiment_setups.append((model, 
                                             model_config,
                                             task,
                                             task_config))
    return experiment_setups


def parse_parenthesized_answers(answers: list, row:pd.Series, raw_choices=None) -> (str, None):
    """
    Returns parsed answer from list pulled from 'response' index of Series. 
    Applies uniqueness presupposed breadth first search a salience ranking (last sentence, antepenultimate + penultimate + last, entire answer) and then through a backoff sequence per salience level as follows:
    exact match substring match, case insensitive substring match and finally
    a case insensitive search with `()` replaced with word boundaries. Has
    ability to use LLM answer parsing but not implemented. 
    Args:
        answers (list): Answers in the form [(A), (B), (C), (D)]
        row (pd.Series): Row form dataframe being evaluated
    Returns: 
        str: Answer if found in original form or None if no answer found
    Raises:
        LookupError if there is not a unique solution 

    Examples Blown UP:
    "Since all the statements (A), (B), (C), and (D) are true, none of them is false. Therefore, there seems to be a mistake in the problem statement or the options provided. However, based on the given options and the analysis, none of the statements is false."

    "So, the answer is:
(A) 0. (B) 1. (C) 2. (D) 3.

The correct answer is:
None of the given options are correct. The dimension that cannot be the dimension of \( V \cap W \) is 4."
        

    """
    matches = set()
    intent_backoff = ['answer is', 'boxed choice', 'exact_word', 'case_insensitive',    
                      'sub_string', 'llm'] #llm unused
    sentences = row['response'].split('\n')
    salience_ranking = []
    salience_ranking.append(sentences[-1:]) # last sentence
    if len(sentences) > 1:
        salience_ranking.append(sentences[-3:]) # up to last 3 sentences
    if len(sentences) > 3:
        salience_ranking.append(sentences) # all sentences
    for sents in salience_ranking:
        text = "\n".join(sents)
        for match_type in intent_backoff:
            for i, answer in enumerate(answers):
                # if match_type == 'boxed choice' and raw_choices not None:
                #     choice = raw_choices[i]
                    
                    #harvest boxed latex
                    #eval and look for float match

                if match_type == 'answer is':
                    escaped_parens = \
                        answer.replace('(', r'\(').replace(')', r'\)')
                    answer_re = fr'(T|t)he answer is {escaped_parens}'
                    if re.search(answer_re, text):
                        matches.add(answer)
                if match_type == 'exact_word':
                    answer_re =\
                            answer.replace('(', r'\(').replace(')', r'\)')
                    if re.search(answer_re, text):
                        matches.add(answer)
                if match_type == 'case_insensitive':
                    answer_re =\
                            answer.replace(')', r'\)').replace('(', r'(?i)\(')
                    if re.search(answer_re, text):
                        matches.add(answer)
                if match_type == 'sub_string':
                    answer_re =\
                        answer.replace(')', r'\b').replace('(', r'(?i)\b')
                    if re.search(fr'{answer_re}', text):
                        matches.add(answer)
            if len(matches) == 1:
                return matches.pop()
            if len(matches) > 1:
                raise LookupError(f"Blown UP: {text}")
        

def parse_string(answers: list, row:pd.Series) -> (str, None):
    """
    Returns parsed answer from list pulled from 'response' index of Series. 
    Applies uniqueness presupposed breadth first search through backoff sequence
    of exact match substring match, case insensitive substring match with word boundaries. Has ability to use LLM answer parsing but not implemented. 
    Args:
        answers (list): Answers in the form ['Yes', 'No']
        row (pd.Series): Row form dataframe being evaluated
    Returns: 
        str: Answer if found in original form or None if no answer found
    Raises:
        LookupError if there is not a unique solution 
    """
    matches = set()
    intent_backoff = ['exact_word', 'case_insensitive', 'llm']
    if not isinstance(row['response'], str):
        return None
    for match_type in intent_backoff:
        for answer in answers:
            if match_type == 'exact_word':
                answer_re = rf'\b{answer}\b'
                if re.search(answer_re, row['response']):
                    matches.add(answer)
            if match_type == 'case_insensitive':
                answer_re = rf'(?i)\b{answer}\b'
                if re.search(answer_re, row['response']):
                    matches.add(answer)
        if len(matches) == 1:
            return matches.pop()
        if len(matches) > 1:
            raise LookupError(f"Blown UP: {row['response']}")


def convert_mmlu_data(data: Dataset):
    """Translates mmlu formatted data from numbers to letters and creates correct format for processing.
    arguments
        data: dict
    returns
        final_data: list of dict
    """
    final_data = []
    option_mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
    for example in data:
        options = "\n"
        for i, option in enumerate(example["choices"]):
            options = options + f"({option_mapping[i]}) {option}. "
        final_data.append(
            {
                "input": f"{example['question']}{options}",
                "target": f'({option_mapping[example["answer"]]})',
            }
        )
    return final_data





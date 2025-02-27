import pandas as pd
from datasets import load_dataset
import os
import sys
sys.path.append(os.getcwd())
import helper_functions


"""
https://huggingface.co/datasets/lukaemon/bbh. 
    
License: MIT License https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/LICENSE
"""

THREE_SHOT = [
    {
        "input": "Which of the following is a humorous edit of this artist or movie name: 'whitesnake'?\nOptions:\n(A) whitesnape\n(B) whitesnapke\n(C) whitesnuake\n(D) mwhitesnake",
        "target": "Let's think step by step.\nThe original name is \"whitesnake\". This is the name of an old English hard rock band. It is a compound word, formed by the words \"white\" and \"snake\".\n(A) \"whitesnape\": It is formed by the combination of \"white\" and \"snake\"; therefore, \"snake\" has been changed to \"snape\". Snape makes a reference to the fictional character Severus Snape in the Harry Potter series, so (A) is indeed a meaningful and funny edit.\n(B) \"whitesnapke\": It is formed by the combination of \"white\" and \"snapke\", but \"snapke\" is not an actual word; therefore, \"whitesnapke\" is not humorous.\n(C) \"whitesnuake\": It is formed by the combination of \"white\" and \"snuake\", but \"snuake\" is not an actual word; therefore, \"whitesnuake\" is not humorous.\n(D) \"mwhitesnake\": It is formed by the combination of \"m\", \"white\", and \"snake\", but the prefix \"-m \"seems arbitrary; therefore, \"mwhitesnake\" is not meaningful or humorous.\nAbove the above, the only humorous edit is (A). So the answer is (A)."
    },

    {
        "input": "Which of the following is a humorous edit of this artist or movie name: 'one of our dinosaurs is missing'?\nOptions:\n(A) ofne of our dinosaurs is missing\n(B) one af our dinosaurs is missing\n(C) one of our dinosaurs is pissing\n(D) one of our dinosaur is missing",
        "target": "Let's think step by step.\nThe original name is \"one of our dinosaurs is missing\". This is the name of an old British movie.\n(A) \"ofne of our dinosaurs is missing\": Here \"one of\" is changed to \"ofne\", but the word \"ofne\" is not an actual word.\n(B) \"one af our dinosaurs is missing\": Here the word \"of\" is changed to \"af\", but the word \"af\" is not an actual word.\n(C) \"one of our dinosaurs is pissing\": Here the word \"missing\" is changed to \"pissing\", and \"one of our dinosaurs is pissing\" is indeed a very whimsical and mischievous edit. This change truly ruins the original title of the movie.\n(D) \"one of our dinosaur is missing\": Here the word \"dinosaurs\" is changed to \"dinosaur\", but \"dinosaur\" is singular but should be plural in the title; this change therefore feels arbitrary and not humorous.\nAbove the above, the only humorous edit is (C).\nAbove the above, the only humorous edit is (C). So the answer is (C)."
    },

    {
        "input": "Which of the following is a humorous edit of this artist or movie name: 'counting crows'?\nOptions:\n(A) countingy crows\n(B) counting cows\n(C) courting crows\n(D) coutnting crows",
        "target": "Let's think step by step.\nThe original name is \"counting crows\". This is the name of an American rock band. Historically, the band name comes from the British nursery rhyme \"One for Sorrow\", which is about counting of magpies.\n(A) \"countingy crows\": Here the word \"counting\" is changed to \"countingy\", but the word \"countingy\" is not an actual word.\n(B) \"counting cows\": Here the word \"crows\" is changed to \"cows\", and this is indeed a playful and meaningful edit that ruins the original name of the band.\n(C) \"courting crows\": Here the word \"counting\" is changed to \"courting\", and \"courting\" is an actual word; however, \"courting crows\" does not sound as humorous as \"counting cows\".\n(D) \"coutnting crows\": Here the word \"counting\" is changed to \"coutnting\", but the word \"coutnting\" is not an actual word.\nAbove the above, the only humorous edit is (B). So the answer is (B)."
    }
]


def get_test_data(config: dict)->list:
    """
    Returns test portion of ruin names task.
    
    Arguments:
        config: dict, 
                {'prompt_type': "v2_style", #return in style of v2 paper
                 'shots': 0}

    Returns:
        list of json text examples
    """
    tasks = list(load_dataset("lukaemon/bbh", name='ruin_names', split="test"))
    if config['prompt_type'] == "v2":
        if config['shots'] == 0:
            return tasks
        if config['shots'] == 'few':
            few_shot_l = \
                [ex["input"] + f" A:{ex['target']}" for ex in THREE_SHOT]
            few_shot = "\n".join(few_shot_l)
            return [{'input': f"{few_shot} {t['input']}", 
                    'target': t['target']} for t in tasks]
            
def raw_fn(row: pd.Series)-> str: 
    """
    Returns raw LLM response. 
    Arguments:
        row: Has raw LLM response in 'response' index
    Returns:
        str
    """
    return row['response']
    
def answer_fn(row: pd.Series, config:dict)-> str: 
    """
    Returns parsed answer from LLM response based on config. 

    Args:
        row (pd.Series): Has raw LLM response in 'response' index
        config (dict): Configuration for the task
    Returns:
        str: Answer in original format or None if no answer found
    Raises:
        LookupError for Blown Uniqueness Presupposition
    """

    if config['prompt_type'] == "v2":
        answers = ['(A)', '(B)', '(C)', '(D)']
        return helper_functions.parse_parenthesized_answers(answers, row)
    
def correct_fn(row: pd.Series, config:dict)-> (bool, None):
    """Determines if system response is correct with Python string
    equivalence. Implements 4 value response, (True, False, None, Blown UP). 
    Args:
        row (pd.Series): Row from task
        config (dict): Configuration for task
    Returns:
        bool: Whether answer is correct, None if no answer
    Raises:
        LookupError for Blown Uniqueness Presupposition
    """
    answer = answer_fn(row, config) #throws LookupError Blown UP
    if answer is None:
        return answer
    return row['gt'] == answer

if __name__ == "__main__":
    import streamlit as st
    ss = st.session_state
    import importlib
    llm = importlib.import_module(f'models.gpt-4o')
    model_config = {"temperature":0.0, "seed": 12, "top_p_k": 0.0}
    st.radio("Shots", options=[0, 'few'], key="shots_r")
    task_config = {'prompt_type': 'v2',
              'shots': ss.shots_r}
    st.write(task_config)
    rubrics = get_test_data(task_config)
    st.slider("Pick Rubric",min_value=0, max_value=len(rubrics), value=0,
              key="rubric_num_sl") 
    st.write(rubrics[ss.rubric_num_sl])
    prompt = [{"role": "user", "content": rubrics[ss.rubric_num_sl]['input']}]
    response, run_config = llm.run(prompt, model_config)
    st.write(response)
    
    





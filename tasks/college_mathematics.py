from datasets import load_dataset
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
import helper_functions
import json
import re


"""
https://huggingface.co/datasets/lukaemon/bbh

MIT License (https://github.com/hendrycks/test/blob/master/LICENSE)


"""

FIVE_SHOT = [
    {
        "input": "Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\n(A) ST = 0 (B) ST = T (C) ST = TS (D) ST - TS is the identity map of V onto itself.",
        "target": "The answer is (D)."
    },
    {
        "input": "Suppose that f(1 + x) = f(x) for all real x. If f is a polynomial and f(5) = 11, then f(15/2)\n(A) -11 (B) 0 (C) 11 (D) 33/2",
        "target": "The answer is (C)."
    },
    {
        "input": "Let A be a real 2x2 matrix. Which of the following statements must be true?\nI. All of the entries of A^2 are nonnegative.\nII. The determinant of A^2 is nonnegative.\nIII. If A has two distinct eigenvalues, then A^2 has two distinct eigenvalues.\n(A) I only (B) II only (C) III only (D) II and III only",
        "target": "The answer is (B)."
    },
    {
        "input": "Let A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?\n(A) -5 (B) -4 (C) -3 (D) -2",
        "target": "The answer is (B)."
    },
    {
        "input": "A tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\n(A) 2 (B) 2 - e^-2 (C) 2 + e^-2 (D) 2 + e^-4",
        "target": "The answer is (D)."
    }
]



def get_test_data(config:dict)->list:
    """
    Returns test portion of dataset. Config 'minimal' gives options and answer
    without much the post-processing needed to get strings.
    Example:

    Config v2 standardizes answers into A-D multiple choice style as done in 
    v1/v2 papers.
    Example:
    

    Returns:
        list of json text examples

    
    """
    dataset = load_dataset("cais/mmlu", name='college_mathematics', 
                            split='test')
    if config['prompt_type'] == "minimal":
        tasks = []
        for task in list(dataset):
                task_d = {'input': 
                  f"{task['question']}\n[{', '.join(task['choices'])}]"}
                task_d['target'] = task['choices'][task['answer']]
                tasks.append(task_d)
        if config['shots'] == 0:
            return tasks
    elif config['prompt_type'] == "v2":
        tasks = list(helper_functions.convert_mmlu_data(dataset))
        if config['shots'] == 0:
            return tasks
        if config['shots'] == 'few':
            few_shot_l = \
                [ex["input"] + f" A:{ex['target']}" for ex in FIVE_SHOT]
            few_shot = "\n".join(few_shot_l)
            return [{'input': f"{few_shot} {t['input']}", 
                    'target': t['target']} for t in tasks]

def raw_fn(row: pd.Series)-> str: 

    """
    Returns raw LLM response. 

    Arguments:
        row: Series, Pandas row, has raw LLM response in 'response' index
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
    try:
        answer = json.loads(row['response'])['Answer'] 
        if answer == row['gt']:
            return True
        gt = re.sub(r'[()]', '', row['gt'])
        if answer == gt:
            return True
        return False
    except json.JSONDecodeError:
        pass
    answer = answer_fn(row, config) #throws LookupError Blown UP
    if answer is None:
        return answer
    return row['gt'] == answer
    
    
def score(config: dict, response: str, truth: str):
    pass

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
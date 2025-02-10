from datasets import load_dataset
import sys
import os
sys.path.append(os.getcwd())
import helper_functions
import pandas as pd


"""
https://huggingface.co/datasets/lukaemon/bbh

MIT License (https://github.com/hendrycks/test/blob/master/LICENSE)


"""



FIVE_SHOT = [
    {
        "input": "Earth Hour was a campaign launched by which organization?\n(A) Greenpeace (B) The UN (C) Oxfam (D) World Wildlife Fund",
        "target": "The answer is (D)."
    },
    {
        "input": "In issues management, what is the most proactive approach to addressing negative or misleading information posted online about your organization?\n(A) Buy domain names that could be used by opposition groups. (B) Post anonymous comments on blogs to combat this information. (C) Prepare a news release that discredits the inaccurate information. (D) Make policy changes to address complaints highlighted on these sites.",
        "target": "The answer is (D)."
    },
    {
        "input": "At which stage in the planning process would a situation analysis be carried out?\n(A) Defining the program (B) Planning the program (C) Taking action and implementing ideas (D) Evaluation of the program",
        "target": "The answer is (A)."
    },
    {
        "input": "Which of these statements is true of the Vatican in 2010 at the time of the accusations of child abuse cover-ups?\n(A) There was a coordinated media response. (B) Consistent messages were communicated. (C) Criticisms were taken as attacks on the Catholic Church. (D) The credibility of the Vatican was upheld.",
        "target": "The answer is (C)."
    },
    {
        "input": "What should a public relations media practitioner do if she does not know the answer to a reporter's question?\n(A) Give the reporter other information she is certain is correct. (B) Say that the information is 'off the record' and will be disseminated later. (C) Say 'I don't know' and promise to provide the information later. (D) Say 'no comment,' rather than appear uninformed.",
        "target": "The answer is (C)."
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
    DATASET = load_dataset("cais/mmlu", name='public_relations', 
                            split='test')
    if config['prompt_type'] == "minimal":
        tasks = []
        for task in list(DATASET):
                task_d = {'input': 
                  f"{task['question']}\n[{', '.join(task['choices'])}]"}
                task_d['target'] = task['choices'][task['answer']]
                tasks.append(task_d)
        if config['shots'] == 0:
            return tasks
    elif config['prompt_type'] == "v2":
        tasks = list(helper_functions.convert_mmlu_data(DATASET))
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
    import sys
    import os

    import helper_functions
    DATASET = load_dataset("cais/mmlu", name='public_relations', 
                            split='test')
    llm = importlib.import_module(f'models.gpt-4o')
    model_config = {"temperature":0.0, "seed": 12, "top_p_k": 0.0}
    st.radio("Shots", options=[0, 'few'], key="shots_r")
    task_config = {'prompt_type': 'v2',
                   'shots': ss.shots_r}
    st.write(task_config)
    rubrics = get_test_data(task_config)
    st.slider("Pick Rubric",min_value=0, max_value=len(rubrics), value=0,
              key="rubric_num_sl") 
    st.write(DATASET[ss.rubric_num_sl])
    st.write(rubrics[ss.rubric_num_sl])
    prompt = [{"role": "user", "content": rubrics[ss.rubric_num_sl]['input']}]
    response, run_config = llm.run(prompt, model_config)
    st.write(response)
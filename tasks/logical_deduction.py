from datasets import load_dataset
import pandas as pd
import helper_functions

THREE_SHOT = [
    {'input': 
"""
The following paragraphs each describe a set of three objects arranged
in a fixed order. The statements are logically consistent within each paragraph.
In a golf tournament, there were three golfers: Amy, Eli, and Eve. Eve finished
above Amy. Eli finished below Amy.

Options:

(A) Amy finished last

(B) Eli finished last

(C) Eve finished last'
target: 'Let''s think step by step.

(1) Eve finished above Amy: "(above) ? Eve ? Amy ? (below)".

(2) Eli finished below Amy: "(above) ? Amy ? Eli ? (below)".

(3) Combining (1) and (2) we get the following ordering: "(above) Eve Amy Eli
(below)".

According to this ordering, the person who finished last (the one at the bottom
of this list) is Eli.

Eli finished last. So the answer is (B).
"""},
    {'input': 
"""The following paragraphs each describe a set of three objects arranged
    in a fixed order. The statements are logically consistent within each paragraph.
    On a shelf, there are three books: a white book, a green book, and an orange
    book. The green book is to the right of the white book. The orange book is the
    rightmost.

    Options:

    (A) The white book is the leftmost

    (B) The green book is the leftmost

    (C) The orange book is the leftmost'
    target: 'Let''s think step by step.

    (1) The green book is to the right of the white book: "(left) ? white ? green
    ? (right)".

    (2) The orange book is the rightmost: "(left) ? white ? green orange (right)".

    (3) Combining (1) and (2) we get the following ordering: "(left) white green
    orange (right)".

    According to this ordering, the leftmost book is the white book.

    The white book is the leftmost. So the answer is (A).'
"""},
{'input': 
"""
The following paragraphs each describe a set of three objects arranged
    in a fixed order. The statements are logically consistent within each paragraph.
    On a shelf, there are three books: a red book, a gray book, and a white book.
    The white book is to the left of the gray book. The red book is the second from
    the left.

    Options:

    (A) The red book is the leftmost

    (B) The gray book is the leftmost

    (C) The white book is the leftmost'
    target: 'Let''s think step by step.

    (1) The white book is to the left of the gray book: "(left) ? white ? gray ?
    (right)".

    (2) The red book is the second from the left: "(left) ? white red gray ? (right)".

    (3) Combining (1) and (2) we get the following ordering: "(left) white red gray
    (right)".

    According to this ordering, the leftmost book is the white book.

    The white book is the leftmost. So the answer is (C).
"""
}]

def get_test_data(config: dict)->list:
    """
    Returns test portion of navigate task.
    
    Arguments:
        config: dict, 
                {'prompt_type': "v2_style", #return in style of v2 paper
                 'shots': 'few'}

    Returns:
        list of json text examples
        Example: {'input': , 'target': }
    """

    tasks = list(load_dataset("lukaemon/bbh",
                 name='logical_deduction_three_objects', 
                            split="test"))
    if config['prompt_type'] == "v2":
        if config['shots'] == 0:
            return tasks
        if config['shots'] == 'few':
            few_shot_l = \
                [ex["input"] for ex in THREE_SHOT]
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

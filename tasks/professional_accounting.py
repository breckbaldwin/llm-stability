from datasets import load_dataset
import helper_functions
import pandas as pd


"""
https://huggingface.co/datasets/lukaemon/bbh

MIT License (https://github.com/hendrycks/test/blob/master/LICENSE)


"""

FIVE_SHOT = [
    {
        "input": "An auditor traces the serial numbers on equipment to a nonissuer\u2019s subledger. Which of the following management assertions is supported by this test?\n(A) Valuation and allocation (B) Completeness (C) Rights and obligations (D) Presentation and disclosure",
        "target": "The answer is (B)."
    },
    {
        "input": "One hundred years ago, your great-great-grandmother invested $100 at 5% yearly interest. What is the investment worth today?\n(A) $13,000 (B) $600 (C) $15,000 (D) $28,000",
        "target": "The answer is (A)."
    },
    {
        "input": "On January 1, year 1, Alpha Co. signed an annual maintenance agreement with a software provider for $15,000 and the maintenance period begins on March 1, year 1. Alpha also incurred $5,000 of costs on January 1, year 1, related to software modification requests that will increase the functionality of the software. Alpha depreciates and amortizes its computer and software assets over five years using the straight-line method. What amount is the total expense that Alpha should recognize related to the maintenance agreement and the software modifications for the year ended December 31, year 1?\n(A) $5,000 (B) $13,500 (C) $16,000 (D) $20,000",
        "target": "The answer is (B)."
    },
    {
        "input": "Krete is an unmarried taxpayer with income exclusively from wages. By December 31, year 1, Krete's employer has withheld $16,000 in federal income taxes and Krete has made no estimated tax payments. On April 15, year 2, Krete timely filed for an extension request to file her individual tax return, and paid $300 of additional taxes. Krete's year 1 tax liability was $16,500 when she timely filed her return on April 30, year 2, and paid the remaining tax liability balance. What amount would be subject to the penalty for underpayment of estimated taxes?\n(A) $0 (B) $500 (C) $1,650 (D) $16,500",
        "target": "The answer is (A)."
    },
    {
        "input": "Box a nongovernmental not-for-profit organization had the following transactions during the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment $10000 Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be reported as net cash provided by financing activities in Box's statement of cash flows?\n(A) $70,000 (B) $75,000 (C) $80,000 (D) 100000",
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
    `{'question': 'You bought a limousine for $98,000 and are planning to rent it for weddings, ceremonies and parties at $245 per hour. If you estimate the car will be hired for 2 hours a day on average, with daily costs at about $50, what is the estimated yearly yield on your investment if you work all year round, i.e. every day of the year, including any festivities and weekends?', 
    'subject': 'professional_accounting', 
    'choices': ['164%', '1.64%', '0.45%', '183%'], 
    'answer': 0}`

    Returns:
        list of json text examples

    
    """
    dataset = load_dataset("cais/mmlu", name='professional_accounting', 
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
    answer = answer_fn(row, config) #throws LookupError Blown UP
    if answer is None:
        return answer
    return row['gt'] == answer
    
    
def score(config: dict, response: str, truth: str):
    pass

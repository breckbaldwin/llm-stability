import os
import sys
import helper_functions
import math
import numpy as np


"""

Model: Simulates stochastic response/answer variation as configured.
Authors: Breck Baldwin.

"""

MODEL = None
MODEL_NAME = "stochastic_dgp"

def run(prompt: list, config: dict) -> (str):
    response = "The answer is A"
    tokens_changed = np.random.binomial(n=config['num_toks'], 
                                        p=config['prob_tok_change'])
    response += f" {tokens_changed}"
    return (response,
            {
            'prompt':prompt, 
            'model_name': MODEL_NAME, 
            'temperature': config['temperature'],
            'seed': config['seed'],
            'top_p_k': config['top_p_k'],
            'rewrite_inst': config.get('rewrite_inst', None),
            'cache_used': False,
            'logprobs': []
            })

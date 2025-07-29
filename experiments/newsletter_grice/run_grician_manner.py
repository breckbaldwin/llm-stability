import os
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
import helper_functions
import run_experiment
from datetime import datetime, date, MINYEAR
import itertools

#MODELS = ['deepseek_R1_0528']
#MODELS = ['llama3-70b', 'gpt-4o_OAI', 'deepseek_R1_0528']

CHOMSKY_PREFIX = "Please answer the following question while adhering to Chomsky’s Competence–Performance distinction: Maximize knowledge (competence) and minimize performance errors. A bit more context,"


GRICE_PREFIX = "Please answer the following question while adhering to Gricean Maxims: Particularly the Maxim of Manner: Be clear—avoid obscurity and ambiguity, be brief and orderly. A bit more context,"

NO_LINGUISTICS = "Please answer the following question,"

COMMON_SUFFIX = " this is a multiple choice question and there is no benefit to the grade in showing your work. You are also being scored on the consistencey of the answer you give over multiple runs so it is suggested you keep your answer to a single letter from A, B, C, D and E and reason conservatively to maximize the chance of giving the same answer across multiple runs of the same question. The question is: "

MODELS = ['gpt-4o_OAI']
MODEL_CONFIGS = [{'temperature': 0.0, 'seed': 12, 'top_p_k': 0.0,
                 #'prefix': NO_LINGUISTICS + COMMON_SUFFIX}
                 'prefix': CHOMSKY_PREFIX + COMMON_SUFFIX}
                 #'prefix': NO_GRICE}
                 ]

TASKS = ['college_mathematics']


TASK_CONFIGS = [{'prompt_type': 'v2', 'shots': 0}]

experiments = itertools.product(MODELS, MODEL_CONFIGS, 
                                                TASKS, TASK_CONFIGS)


for model, model_config, task, task_config in experiments:
    date = datetime.now()
    datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")
    #task_config_in_filename = 'chomsky'
    task_config_in_filename = 'str_chomsky'
    
    run_args = {'output_directory': 'runs/',
                'model': model,
                'model_config': model_config,
                'model_config_in_filename': model_config['temperature'],
                'task': task,
                'task_config': task_config,
                'task_config_in_filename': task_config_in_filename,
                'num_runs': -1,
                'end_on_answer': False,
                'keep_disagreement': True,
                #'limit_num_rubrics': 2
    }
#    datetime_string = '0001-01-01_00-00-00'
    val = run_experiment.run(run_args, datetime_string)
    if val == "Successfully run":
        break
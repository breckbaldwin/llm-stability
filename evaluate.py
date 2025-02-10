import itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
import pandas
import argparse
import glob
import json
from collections import defaultdict
from typing import Callable
import importlib
from scipy.stats import bootstrap
import random
import re

sys.path.append(os.getcwd())

"""
Evaluation for code for most experiments. 

Runs a given experiment as configured by command line parameters. 

@authors Breck Baldwin

 * Copyright 2024 Comcast Cable Communications Management, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
"""

MODELS =['gpt-4o', 'llama-3-70b', 'finetuned-3.5', 'gpt-3.5-turbo',
        'mixtral-8x7b', 'llama-3-8b']
TASKS = ['high_school_european_history', 'college_mathematics',
         'geometric_shapes', 'navigate', 'professional_accounting', 
         'logical_deduction', 'ruin_names', 'public_relations']
SHOTS = ['0-shot', 'few_shot']

def bootstrap():
    import numpy as np
    import math
    from helper_functions import parse_file_name

    ci_95 = bootstrap((data,), np.mean, n_resamples=2000, 
                      confidence_level=0.95,
            random_state=7)
    low, high = ci_95.confidence_interval
    if math.isnan(low):
        low, high = np.mean(data), np.mean(data)
    print(f"Model: {model}, Task: {task}, 95% CI: ({low=:.3f},{high=:.3f}), Mean: {np.mean(data)=:.3f}")


def load_runs(directory, old_format=False) -> pd.DataFrame:
    """
    Loads all .csv files in indicated directory recursively and expands likely
    serialized dicts from their respective columns. 
    Arguments:
        args: str, dir with .csv files
        old_format: bool, if True convert old format files to new
    Returns:
        pd.DataFrame: Files concatenated into a df
    """
    files = glob.glob(os.path.join(directory, '**', '*.csv'), 
                        recursive=True)
    data_df = pd.DataFrame()
    
    for csv_file in files:
        csv_df = pd.read_csv(csv_file)
        print(f"Trying {csv_file}")
        if old_format:
            model = None
            for model_name in MODELS:
                if model_name in csv_file:
                    model = model_name
            
            assert model is not None
            csv_df['model'] = 'gpt-35-turbo' if model == 'gpt-3.5-turbo' \
                                else model
            model_config = {"temperature": 0.0, 
                            "seed": 12, 
                            "top_p_k": 0.0 if 'p0' in csv_file else 1.0}
            csv_df['model_config'] = json.dumps(model_config)
            task = None
            for task_name in TASKS:
                if task_name in csv_file:
                    task = task_name
            assert task is not None
            csv_df['task'] = task
            shot = None
            for shot_name in SHOTS:
                if shot_name in csv_file:
                    shot = shot_name if shot_name != 'few_shot' else 'few'
            assert shot is not None
            task_config = {"prompt_type": "v2", "shots": shot}
            csv_df['task_config'] = json.dumps(task_config)
            csv_df['prompt'] = '{}'
            csv_df['rubric'] = '{}'
            csv_df['rubric_id'] = csv_df.index
            csv_df['response'] = csv_df['raw_response']
            m = re.match(r'.*_(\d+)\.csv', csv_file)
            csv_df['run'] = int(m.group(1))
        for col in ['prompt', 'model_config', 'task_config', 'rubric']:
            csv_df[col] = csv_df[col].apply(lambda row: json.loads(row))
        csv_df['file'] = csv_file
        data_df = pd.concat([data_df, csv_df], ignore_index=True)
    return data_df


def get_experiment_configs(data_df: pd.DataFrame)-> list:
    """
    Returns all possible combinations of experiment configs, does
    not check that configs exist together in data_df. 
    Args:
        data_df (pd.DataFrame): Experiment runs 
    Returns:
        list: Tuples (model, model_config, task, model_config)
    """
    exp_configs = []
    for model in data_df['model'].unique():
        for model_config in data_df['model_config'].drop_duplicates():
            for task in data_df['task'].unique():
                for task_config in data_df['task_config'].drop_duplicates():
                    exp_configs.append((model, 
                                        model_config, 
                                        task, 
                                        task_config))
    return exp_configs

            
def evaluate(data_df: pd.DataFrame, num_bootstrap_draws=10) -> (dict, pd.DataFrame, list):
    """
    Runs counting and computation of TARa, TARr and correct. Organized by
    model x model_config x task x task_config.  
    Unit tests in `tests/test_evaluate.py` show simple examples. 

    Arguments:
        data_df: dataframe, will be iterated over by the 4-tuple model x model_config x task x task_config. Assumes only one collection of runs 
        exists per designated 4-tuple. 
    Returns:
        pd.DataFrame: Columns are: 'model',
                                    'model_config',
                                    'task',
                                    'task_config',
                                    'TACr',
                                    'TARr,
                                    'TACa',
                                    'TARa',
                                    'correct_count_per_run',
                                    'correct_pct_per_run',
                                    'num_questions',
                                    'N'
    """ 
    
    data_df['correct'] = False
    data_df['parsed_answer'] = None
    configs = get_experiment_configs(data_df)
    results = pd.DataFrame()
    errors = []
    seen_errors = defaultdict(int)
    total_evals = 0
    task_x_rubric = set()
    
    for model, model_config, task, task_config in configs:
        try:
            task_module = importlib.import_module(f'tasks.{task}')
        except ModuleNotFoundError:
            print(f'Need to add {f"tasks.{task}"}, skipping eval')
            continue
        exp_df = data_df[(data_df['model'] == model)
                        & (data_df['model_config'] == model_config)
                        & (data_df['task'] == task)
                        & (data_df['task_config'] == task_config)]
        print(f"{model} {model_config} {task} {task_config}")
        if len(exp_df.index) == 0: #may have combos with no data
            continue
        total_agreement_count_raw = 0
        total_agreement_count_answer = 0
        num_runs = max(exp_df['run']) + 1
        correct = [0] * num_runs # need to track correct for run number
        any_correct_count = 0
        any_wrong_count = 0
        rubric_ids = exp_df['rubric_id'].unique()
        num_questions = len(rubric_ids)
        runs_accum = []
        for id in rubric_ids:
            task_x_rubric.add(f"{task}x{id}")
            question_df = exp_df[exp_df['rubric_id'] == id]
            #question_df = question_df.reset_index(drop=True)
            raw = set()
            answer = set()
            if not num_runs == len(question_df.index):
                print(f"{model}, {model_config}, {task}, {task_config}")
                error = f"runs not matching expected length, expected {num_runs}, got {len(question_df.index)} for {question_df['file'].to_list()}"
                raise IndexError(error)
            corrects_for_rubric = 0
            run_accum = [0] * num_runs
            for idx, row in question_df.iterrows(): # runs over question
                raw.add(task_module.raw_fn(row))
                total_evals += 1
                if pd.isna(row['response']):
                    errors.append(f"NaN response found {model} {task} {id}")
                    continue
                try:
                    parsed_answer = task_module.answer_fn(row, task_config)
                    if parsed_answer is None :
                        error = f"No answer found {model} {task} {id}"
                        seen_errors[error] += 1
                        if seen_errors[error] == 1:
                            print(f"-----------------parse issue---{len(errors)}")
                            print(f"{error} {row['run']}")
                            print(f"Response: {row['response']}")
                            print(f"Rubric: {row['rubric']}")

                            errors.append(f"No answer found {model} {task} {id} {row['run']}")
                            answer.add(idx) # No answer will always fail TARa
                        print(f"Repeat {seen_errors[error]} for rubric id {id}")
                        errors.append(f"No answer found {model} {task} {id} {row['run']}")
                        continue # cannot be correct so continue
                    else:
                        data_df.loc[idx,'parsed_answer'] = parsed_answer
                        answer.add(parsed_answer)
                except LookupError as e:
                    answer.add(idx) # Blown UP is also a failure of TARa
                    data_df.loc[idx,'parsed_answer'] = "Blown UP"
                    error = f"Blown UP found {model} {task} {id}"
                    seen_errors[error] += 1
                    if seen_errors[error] == 1:
                        print(f"------answer issue-----{len(errors)}---")
                        print(f"{error} {row['run']}")
                        print(f"Response: {row['response']}")
                        print(f"Rubric: {row['rubric']}")
                    errors.append(f"Blown UP {model} {task} {id} {row['run']}")
                    print(f"Repeat {seen_errors[error]} for rubric id {id}")
                    continue # can't be correct so continue
                if task_module.correct_fn(row, task_config):
                    correct[row['run']] += 1
                    run_accum[row['run']] = 1
                    data_df.loc[idx,'correct'] = True
                    corrects_for_rubric += 1
            if len(raw) == 1:
                total_agreement_count_raw += 1
            if len(answer) == 1:
                total_agreement_count_answer += 1
            if corrects_for_rubric != num_runs:
                any_wrong_count += 1
            if corrects_for_rubric > 0:
                any_correct_count += 1
            runs_accum.append(run_accum)
            # for N iterations, draw
        bootstrap_correct_counts = [0] * num_bootstrap_draws
        for d in range(num_bootstrap_draws):
            for run in runs_accum:
                bootstrap_correct_counts[d] += random.choice(run)
        bootstrap_pct = [c/num_questions for c in bootstrap_correct_counts]

        result_d = {'model': model, 
                    'model_config': model_config,
                    'task': task,
                    'task_config': task_config,
                    'TACr': total_agreement_count_raw,
                    'TARr': total_agreement_count_raw/num_questions,
                    'TACa': total_agreement_count_answer,
                    'TARa': total_agreement_count_answer/num_questions,
                    'correct_count_per_run': correct,
                    'correct_pct_per_run': [c/num_questions for c in correct],
                    'num_questions': num_questions,
                    'N': num_runs,
                    'best_possible_count': any_correct_count,
                    'best_possible_accuracy': any_correct_count/num_questions,
                    'worst_possible_count': num_questions-any_wrong_count,
                    'worst_possible_accuracy': (num_questions-any_wrong_count)/num_questions,
                    'spread': any_correct_count/num_questions - 
                              (num_questions-any_wrong_count)/num_questions,
                    'bootstrap_counts': sorted(bootstrap_correct_counts),
                    'bootstrap_pcts': sorted(bootstrap_pct),
                    'date': exp_df['date'].iloc[0]
        }
        result_s = pd.Series(result_d)
        results = \
            pd.concat([results, result_s.to_frame().T],  ignore_index=True)
        #print("ERRORS:")
        #print(errors)
        
        print((f"{len(seen_errors.keys())} rubrics had parsing problems for"
        + f" {len(task_x_rubric)} task x rubrics for {total_evals} total"
        + f" evaluations"))
    return results, data_df, errors
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", required=True, 
                            help=("path to data dir to timestamp, e.g., " 
        + "experiments/low_temp/local_runs/10_tasks/2024-11-29_13-49-06/"))
    parser.add_argument("-eo", "--eval_orig", action="store_true",
                        required=False,
                        default=False,
                        help="Evaluate old run format from v2 paper")
    #parser.add_argument("-g", "--graph_location", required=True,
    #                    help="path/filename.png to write graph to. ")
    command_args = parser.parse_args()
    data_df = load_runs(command_args.directory, command_args.eval_orig)
    
    (eval_df, dict, errors) = evaluate(data_df)
    print(eval_df)
    eval_df.to_csv("evaluation_output.csv")

    #graph(epsilons, diffs, args.graph_location)
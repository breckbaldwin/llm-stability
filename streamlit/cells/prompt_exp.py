import streamlit as st
import sys
import os
#from mlflow import log_metric, log_param, log_params, log_artifacts
#import datasets
import itertools
import pandas as pd
from collections import defaultdict
import json
sys.path.append(os.path.join(os.getcwd(), "cells"))
sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.path.join(os.getcwd(), "../tasks"))
sys.path.append(os.path.join(os.getcwd(), "../models"))
import cells.cell_util

from typing import Callable
import importlib
from datetime import datetime, date, MINYEAR

import run_experiment
import evaluate

ss = st.session_state
#save to config
# rerun and keep count

def setup_globals(page, i_cell):
    pass

json_prompt = """
Please answer the following question adhering to these format instructions:
The output should be formatted as a JSON instance that conforms to the JSON schema below.
 
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "Answer": {
      "type": "string",
      "enum" : ["A", "B", "C", "D"]
    }
  },
  "required": [
    "Answer"
  ]
}

The output {"Answer": "A"} is a well-formatted instance of the schema, the output {"Answer": "E"} is not well-formatted. A string answer like "The correct answer is A" is not well-formatted.

The question is: 


"""

def update_setup_from_widget(key, ss_widget, reload):
     ss.setup[key] = ss[ss_widget] 
     if reload:
        ss.exp_loaded = False
        ss.eval_df = None

def update_setup_config_from_widget(key, sub_key, ss_widget):
    ss.setup[key][sub_key] = ss[ss_widget] 
    ss.exp_loaded = False
    ss.eval_df = None


def wipe_test_data():
    ss.test_data = None
    ss.eval_df = None

def format(prefix, question, suffix):
    return "\n".join([ele for ele in [prefix, question, suffix] if ele != ''])

def run_ui(page, i_cell):
    ss.setup = ss.config[page][i_cell]['setup']
    if 'eval_df' not in ss:
        ss.eval_df = None
    Comms = st.expander("Communcations/Credits")
    Comms.write("""

Contact: Breck Baldwin (breckbaldwin@comcast.com, breckbaldwin@slack)

Suggestions/Issues/Software: Put an issue at: https://github.com/comcast-explainable-ai-lab-group/llm_stability

Project info: 

- Comcast Repo: https://github.com/comcast-explainable-ai-lab-group/llm_stability/blob/main/README.md

- Public Repo: (out of date) https://github.com/Comcast/llm-stability 

Publication(s): https://arxiv.org/abs/2408.04667

The gang: 

Teams channel: https://teams.microsoft.com/l/channel/19%3AezGJaJC1xqJmU4mUwffvWTU4m5DNikHvqlqpNmgHUjg1%40thread.tacv2/LLM-Lab%20Channel?groupId=859e339e-141f-4b15-b2c1-fa2d717133dc&tenantId=906aefe9-76a7-4f65-b82d-5ec20775d5aa

Channel email: LLM-Lab Channel - LLM Lab <7d9916e1.comcastcorp.onmicrosoft.com@amer.teams.ms>

Members:
- Baldwin, Breck
- Walker Jr., John
- Chokhawala, Bhagyeshkumar
- Sloan, Adam
- Wu, Zhe
- Riviello, John
- Hansen-Turton, Brian
- Aykent, Sarp
- Tudrej, Tomasz
- Radcliffe, Evan
- Gupta, Deepti
- Rajagopal, Guru Rajan

    """)
    Admin = st.expander("Admin")
    video = Admin.file_uploader("Upload Instruction Video", type=['mov'])
    if video is not None:
        bytes_data = video.getvalue()
        with open("Demo.mov", mode="wb") as file:
            file.write(bytes_data)
    Admin.checkbox("Advanced Tasks", value=False, key="advanced_cb")
    if st.checkbox("Load Video"):
        video_file = open("Demo.mov", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes, format='video/quicktime')
    
    Setup_cols = st.columns(5)
    model_options = ['', #"llama-3-70b", "mixtral-8x7b-instruct", 
                #"finetuned-3.5", 
                'gpt-35-turbo',
                "gpt-4o"]
    Setup_cols[0].selectbox("Model", options=model_options,
                            index=model_options.index(ss.setup['model']),
                            on_change=update_setup_from_widget,
                            args=['model', 
                                  f"model_name_{ss.odd_even}_sb", 
                                  True],
                            key=f"model_name_{ss.odd_even}_sb")
    lukaemon_bbh = ['professional_accounting', 'geometric_shapes', 'logical_deduction', 'geometric_shapes']
    cais_mmlu = ['navigate', 'college_mathematics', 
                 'high_school_european_history', 'public_relations']
    task_options = lukaemon_bbh + cais_mmlu
    Setup_cols[1].selectbox("Task", 
                            options=task_options,
                            index=task_options.index(ss.setup['task']),
                            on_change=update_setup_from_widget,
                            args=['task', 
                                  f"task_{ss.odd_even}_sb",
                                  True],
                            key=f"task_{ss.odd_even}_sb")
    if ss.advanced_cb:
        Setup_cols[2].slider("Temperature", min_value=0.0, max_value=1.0, 
                        value=ss.setup['model_config']['temperature'],
                        on_change=update_setup_config_from_widget,
                        args=['model_config', 
                              'temperature', 
                              f"temp_{ss.odd_even}_sl"],
                        key=f"temp_{ss.odd_even}_sl")
        format_options = ['v2']
        max_shots_for_data = 5
        if ss.setup['task'] in lukaemon_bbh:
            format_options = ['v2', 'minimal']  
            max_shots_for_data = 3
        Setup_cols[3].selectbox("Question format", 
                             options=format_options,
                             index=\
                                format_options.index(ss.setup['task_config']['prompt_type']),
                             on_change=update_setup_config_from_widget,
                             args=['task_config',
                                    'prompt_type', 
                                    f"format_{ss.odd_even}_sb"],
                             key=f"format_{ss.odd_even}_sb")
        shots_options = list(range(max_shots_for_data + 1))
        Setup_cols[4].selectbox("Shots", 
                         options=shots_options,
                         index=\
                          shots_options.index(ss.setup['task_config']['shots']),
                         on_change=update_setup_config_from_widget,
                         args=['task_config', 
                                'shots', 
                                f"shots_{ss.odd_even}_sb"],
                         key=f"shots_{ss.odd_even}_sb")

    if not ss.exp_loaded:
        task_module = importlib.import_module(ss.setup['task'])
        task_config = ss.setup['task_config']
        ss.test_rubrics = task_module.get_test_data(task_config)
        ss.test_rubrics_df = pd.DataFrame(ss.test_rubrics)
        ss.llm = importlib.import_module(ss.setup['model'])
        ss.rubric_id = 0
        ss.exp_loaded = True
            
    # The editing UI
    #st.dataframe(ss.test_rubrics_df)    
    st.text_area("Prefix:", value=ss.setup["prefix_prompt"], key="prefix_ta")
    st.markdown(f"Number: {ss.rubric_id}")
    st.json({'Question:': ss.test_rubrics[ss.rubric_id]['input']})
    st.text_area("Suffix:", value=ss.setup["suffix_prompt"], key="suffix_ta")

    Run_cols = st.columns(4)
    Run_cols[1].slider("Repeat count: ", 
                min_value=1, 
                max_value=10, 
                value=ss.setup['N'],
                on_change=update_setup_from_widget,
                         args=['N', f"N_{ss.odd_even}_sl", False],
                key=f"N_{ss.odd_even}_sl")
    Run_cols[2].slider("Number of rubrics to run:", min_value=1, 
                        max_value=len(ss.test_rubrics_df.index), 
                        value=1, key="number_to_run_s")
    if 'run' not in ss:
        ss.run=False

    if Run_cols[0].button("Run"):
        ss.run = True
        if 'run_df' in ss:
            del ss.run_df
        if 'eval_df' in ss:
            del ss.eval_df

    #if Run_cols[2].button("Increment and Run"):
    #    ss.index +=1
    #    ss.run = True
    #    ss.counter = 0
    #Run_cols[3].checkbox("Show Prompt:", value=False, key="show_prompt_sb")
    #Run_cols[3].checkbox("Show Answer:", value=False, key="show_answer_sb")
    #Run_cols[3].checkbox("Show Truth:", value=False, key="show_truth_sb")
    
    if ss.run:
        ss.run = False
        username = 'none'
        date = datetime.now()
        datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")
        out_date_dir = os.path.join('ui_runs', username, datetime_string)
        print(f"Making output directories if necessary {out_date_dir}")
        os.makedirs(out_date_dir, exist_ok=True)
        outfile_root = (f"{ss.setup['model']}"
                      + f"-{ss.setup['model_config']['temperature']}"
                      + f"-{ss.setup['task']}"
                      + f"-{ss.setup['task_config']['shots']}")
        ss.pref_suff_rubrics = []
        for rubric in ss.test_rubrics[:ss.number_to_run_s]:
            pref_suff_rubric =\
                 {'input': format(ss.prefix_ta, rubric['input'], ss.suffix_ta),
                  'target': rubric['target']}
            ss.pref_suff_rubrics.append(pref_suff_rubric)
        run_experiment.run_model(
                            llm=ss.llm,
                            out_dir=out_date_dir, 
                            outfile_root=outfile_root, 
                            num_runs=ss.setup['N'],
                            model_config=ss.setup['model_config'], 
                            test_rubrics=ss.pref_suff_rubrics,
                            model_name=ss.setup['model'],
                            task_name=ss.setup['task'],
                            task_config=ss.setup['task_config'],
                            date=datetime_string,
                            context=st)
        ss.run_df = evaluate.load_runs(os.path.join(out_date_dir))
        
        ss.eval_df, ss.data_df, ss.errors = evaluate.evaluate(ss.run_df)
    if ss.eval_df is None:
        st.stop()
    
    Eval_cols = st.columns([2,1,1])
    display_df = ss.eval_df.copy()
    Eval_cols[0].multiselect("Select Eval Columns", options=display_df.columns, key="eval_cols_ms")
    Eval_cols[1].radio("Include/exclude", options=['include', 'exclude'], 
                        index=1,
                        label_visibility="hidden", key="inc_excl_eval_r")
    display_df['correct_pct_per_run'] =\
       display_df['correct_pct_per_run'].apply(lambda x: [f"{val:.1%}" for val in x])
    cols = ss.eval_cols_ms
    if ss.inc_excl_eval_r == 'exclude':
        cols = [c for c in display_df.columns if c not in ss.eval_cols_ms]
    display_df = display_df[cols]
    st.dataframe(display_df.style.format({'TARa':'{:.1%}', 'TARr':'{:.1%}'}))
    
    run_display_df = ss.data_df.copy()
    run_display_df['prompt'] = \
        run_display_df['prompt'].apply(lambda x: json.dumps(x, indent=4))
    Run_cols = st.columns([2, 1, 1, 2])
    Run_cols[0].multiselect("Select Run Columns", options=run_display_df.columns, key="run_cols_ms")
    Run_cols[1].radio("Include/exclude", options=['include', 'exclude'], 
                        index=1,
                        label_visibility="hidden", key="inc_excl_run_r")
    cols = ss.run_cols_ms
    if ss.inc_excl_run_r == 'exclude':
        cols = [c for c in run_display_df.columns if c not in ss.run_cols_ms]
    
    Run_cols[3].multiselect("Error Search", 
                      options=['raw disagreement across runs', 
                               'parsed_answer disagreement across runs', 
                               'parsed_answer: All None', 
                               'parsed_answer: All Blown UP',
                               'incorrect parsed answer'], 
                      key="error_search_ms") 
    if Run_cols[2].button("Find Mismatch"):
        if 'search_id' not in ss:
            ss.search_id = 0
        
        while ss.search_id != len(ss.pref_suff_rubrics):
            print(f"Trying {ss.search_id}")
            if ss.search_id % 10 == 0:
                Run_cols[2].markdown(f"Tried to {ss.search_id}")
            rubric_df = run_display_df[run_display_df['rubric_id'] \
                                        == ss.search_id]
            ss.search_id += 1
            error_found = False
            if 'raw disagreement across runs' in ss.error_search_ms:
                error_found = len(rubric_df['response'].unique()) > 1
            if 'parsed_answer disagreement across runs' in ss.error_search_ms:
                error_found = len(rubric_df['parsed_answer'].unique()) > 1 
            if  'parsed_answer: All None' in ss.error_search_ms:
                error_found = rubric_df['parsed_answer'].iloc[0] is None
            if 'parsed_answer: All Blown UP' in ss.error_search_ms:
                error_found = rubric_df['parsed_answer'].iloc[0] ==\
                    'Blown UP'    
            if 'incorrect parsed answer' in ss.error_search_ms:
                error_found = len(rubric_df['correct'].unique()) > 1
                error_found = not rubric_df['correct'].iloc[0]
            if error_found:
                st.dataframe(rubric_df[cols])
                break
        if ss.search_id == len(ss.pref_suff_rubrics):
            st.info("Search to the end, press 'Find Mismatch' again to restart")
            ss.search_id = 0

            
        
    if st.checkbox("Show All Results"):
        run_display_df = run_display_df[cols]
        st.dataframe(run_display_df)   

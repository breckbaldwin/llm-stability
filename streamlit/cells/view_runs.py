import streamlit as st
import sys
import os
import pandas as pd
from collections import defaultdict
import json
sys.path.append(os.path.join(os.getcwd(), "cells"))
import cells.cell_util
import re
import glob

ss = st.session_state

def load_data():
    if ss.exp_dir_sb == '':
        st.info("Please select a directory of runs")
        return
    if ss.variation_sb == '':
        st.info("Please select a run to view")
        return
    if ss.file_sb == '':
        st.info("Please select a file")
        return
    if ss.all_runs_cb:
        m = re.match(r'(.*)_(\d+).csv', ss.file_sb)
        if m:
            root = m.group(1)
            files = glob.glob(os.path.join("..",ss.exp_dir_sb, 
                                            ss.variation_sb, f'{root}_*.csv'))
            accum = []
            for file in files:
                if not re.match(r'(.*)_(\d+).csv', file): #glob can over generate
                    continue
                m = re.match(r'(.*)(\d+).csv', file)
                run_number = m.group(2)
                df = pd.read_csv(file)
                df['run'] = run_number
                accum.append(df)
            ss.view_df = pd.concat(accum)
 #           ss.view_df = ss.view_df.drop(['pred', 'Unnamed: 0'], axis=1)
            print(ss.view_df.columns)
    else:
        breakpoint()
        ss.view_df = pd.read_csv(os.path.join("..",ss.exp_dir_sb, 
                                              ss.variation_sb, ss.file_sb))

#        ss.view_df = ss.view_df.drop(['pred', 'Unnamed: 0'], axis=1)

def run_ui(page, cell_i):
    if 'run_to_view' not in ss:
        ss.run_to_view = ''
    st.markdown("View Runs")
    setup = ss.config[page][cell_i]['setup']
    options = [''] + setup['experiment_dirs']
    try:
        idx1 = options.index(setup['selected_dir'])
    except ValueError as e:
        idx1 = 0
    st.selectbox("Select experiment dir", options=options,
                 index=idx1, key="exp_dir_sb")
    dirs = [''] + os.listdir(os.path.join("..", ss.exp_dir_sb))
    try:
        indx = dirs.index(setup['variation'])
    except ValueError as e:
        indx = 0
    st.selectbox("Select variation", options=dirs, 
                    index=indx, 
                    key="variation_sb")
    variation_files = [''] + os.listdir(os.path.join("..", ss.exp_dir_sb, 
                                                ss.variation_sb))
    st.selectbox("Select file", options= variation_files, 
                    index=variation_files.index(setup['run_to_view']),
                    on_change=load_data,
                    key="file_sb"
                )
    st.checkbox("Bring in all runs", value=True, key="all_runs_cb")
    if 'view_df' not in ss:
        load_data()
    if 'view_df' in ss:
        st.dataframe(ss.view_df)
    
    


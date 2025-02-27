import streamlit as st
import pandas as pd
import os
import json



def run():
    st.title("CSV Browser")

    df = pd.read_csv("../local_runs/gpt-35-turbo-0.0-college_mathematics-0_2025-02-08_16-51-24/gpt-35-turbo-0.0-college_mathematics-0-0.csv")
    
    
    for i, row in df.iterrows():
        if json.loads(row['prompt'])[0]['content'] != \
            json.loads(row['rubric'])['input']:
            st.markdown(i)
            st.markdown(row['prompt']['content'])
            st.markdown(row['rubric']['input'])
    #st.dataframe(df[['prompt', 'rubric']])


    




if __name__ == '__main__':
    run()
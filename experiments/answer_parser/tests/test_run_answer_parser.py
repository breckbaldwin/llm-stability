import pytest
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
import run_answer_parser




def test_parse_responses():
    data_df = pd.DataFrame({
                            'task': ['college_mathematics'] * 3,
                            'task_config': [{}] * 3,
                            'rubric_id': [1, 1, 1],
                            'rubric': [{'input': '', 'target': '(A)'}] * 3,
                            'response': ['(A)', '(A)', '(C)'],
                            'gt': ['(A)'] * 3
                            })
    result_df = run_answer_parser.parse_responses(data_df)
    assert len(result_df.index) == 2
    assert result_df[result_df['response'] == '(A)'].iloc[0]['count'] == 2
    assert result_df[result_df['response'] == '(C)'].iloc[0]['count'] == 1
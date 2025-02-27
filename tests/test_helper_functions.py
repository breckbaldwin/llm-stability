import pytest
import os
import sys
sys.path.append(os.getcwd())
import helper_functions
import pandas as pd


def test_parse_parenthesized_answers():
    answers = ['(A)', '(B)']
    row = pd.Series({'response': 'The answer is (A)'})
    #answer = helper_functions.parse_parenthesized_answers(answers, row)
    #assert answer==answers[0]
    row['response'] = "The answer is (A). \n The answer is (B)."
    answer = helper_functions.parse_parenthesized_answers(answers, row)
    assert answer == answers[1]
    with pytest.raises(LookupError) as e:
        row['response'] = "The answer is (A), the answer is (B)."
        answer = helper_functions.parse_parenthesized_answers(answers, row)
        assert e.value == "Blown UP: The answer is (A). The answer is (B)."
    
    row['response'] = "The answer is (B). \n Some Noise."
    answer = helper_functions.parse_parenthesized_answers(answers, row)
    assert answer == answers[1]

    with pytest.raises(LookupError) as e:
        row['response'] = "The answer is (A). \n The answer is (B). \nSome noise."
        answer = helper_functions.parse_parenthesized_answers(answers, row)
        assert e.value == "Blown UP: The answer is (A). \n The answer is (B). \nSome noise."
 



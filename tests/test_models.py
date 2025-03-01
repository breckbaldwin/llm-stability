import pytest
import os
import sys
sys.path.append(os.getcwd())
import tasks.professional_accounting
import tasks.navigate
import importlib
import json

"""
Testing models for basic processing and authentication. Usage:

`pytest tests/test_models.py`

To test single model, e.g., gpt-35-turbo:

`pytest tests/test_models.py::test_gpt_35_turbo`


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

def test_gpt_35_turbo():
    model_name = 'gpt-35-turbo'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

    
def test_gpt_4o():
    model_name = 'gpt-4o'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

def test_llama_8b():
    model_name = 'llama3-8b'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    #assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

def test_llama_70b():
    model_name = 'llama3-70b'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    #assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

def test_mixtral_8x7b():
    model_name = 'mixtral-8x7b'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but need to run a unit test."}] #be polite to our overlords
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0})
    assert run_info['prompt'] == test_prompt
    #assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert len(result) > 10

def test_gpt_4o_rewrite():
    model_name = 'gpt-4o'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    rewrite_inst = ("Please rewrite the below prompt to maximize your"
                      + " understanding of the instruction for processing.")
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'rewrite_inst': rewrite_inst})
    assert len(run_info['prompt'][0]['content']) > 10
    assert run_info['prompt'] != test_prompt
    assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert run_info['rewrite_inst'] == rewrite_inst
    assert len(result) > 10
    assert not run_info['cache_used']

    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'rewrite_inst': rewrite_inst})
    assert run_info['cache_used']


def test_gpt_3_5_turbo_rewrite():
    model_name = 'gpt-35-turbo'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    rewrite_inst = ("Please rewrite the below prompt to maximize your"
                      + " understanding of the instruction for processing.")
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'rewrite_inst': rewrite_inst})
    assert len(run_info['prompt'][0]['content']) > 10
    assert run_info['prompt'] != test_prompt
    assert run_info['model_name'] == model_name
    assert run_info['temperature'] == 1.0
    assert run_info['seed'] == 13
    assert run_info['top_p_k'] == 0.0
    assert run_info['rewrite_inst'] == rewrite_inst
    assert len(result) > 10
    assert not run_info['cache_used']

    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'rewrite_inst': rewrite_inst})
    assert run_info['cache_used']

def test_gpt_3_5_turbo_prefix_suffix():
    model_name = 'gpt-35-turbo'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    prefix = "x x x x x x "
    suffix = " y y y y y"
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'prefix': prefix,
                                            'suffix': suffix})
    assert run_info['prompt'][0]['content'] ==\
         f"{prefix}{test_prompt[0]['content']}{suffix}"

def test_gpt_4o_prefix_suffix():
    model_name = 'gpt-4o'
    llm = importlib.import_module(f'models.{model_name}')
    test_prompt = [{"role": "user", "content": "Wakey wakey,u up? Sorry to bother you but can you run this unit test."}] #be polite to our overlords
    prefix = "x x x x x x "
    suffix = " y y y y y"
    result, run_info = llm.run(test_prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'prefix': prefix,
                                            'suffix': suffix})
    assert run_info['prompt'][0]['content'] ==\
         f"{prefix}{test_prompt[0]['content']}{suffix}"


def test_gpt_4o_OAI():
    system_prompt = 'Please answer the following question with the answer field only.'
    schema = {
            "name": "yes_no",
            "schema":{ 
                "type": "object",
                "properties": {
                    "Answer": {
                    "type": "string",
                    "enum" : ["Yes", "No"]
                    }
                },
                "required": [
                    "answer"
                ]
            }
        }
    model_name = 'gpt-4o_OAI'
    llm = importlib.import_module(f'models.{model_name}')
    prompt = [{"role": "user", "content": "Is 835 even?"}]
    result, run_info = llm.run(prompt, {'temperature':1.0,
                                            'seed': 13,
                                            'top_p_k': 0.0,
                                            'system_content': system_prompt,
                                            'answer_schema': schema})
    assert len(run_info['prompt'][0]['content']) > 0
    answer_d = json.loads(result)
    assert answer_d['Answer'] == 'No'







    #math_tutor_prompt = '''
#     You are a helpful math tutor. You will be provided with a math problem,
#     and your goal will be to output a step by step solution, along with a final answer.
#     For each step, just provide the output as an equation use the explanation field to detail the reasoning.
# '''

# def get_math_solution(question):
#     response = client.chat.completions.create(
#     model=MODEL,
#     messages=[
#         {
#             "role": "system", 
#             "content": dedent(math_tutor_prompt)
#         },
#         {
#             "role": "user", 
#             "content": question
#         }
#     ],
#     response_format={
#         "type": "json_schema",
#         "json_schema": {
#             "name": "math_reasoning",
#             "schema": {
#                 "type": "object",
#                 "properties": {
#                     "steps": {
#                         "type": "array",
#                         "items": {
#                             "type": "object",
#                             "properties": {
#                                 "explanation": {"type": "string"},
#                                 "output": {"type": "string"}
#                             },
#                             "required": ["explanation", "output"],
#                             "additionalProperties": False
#                         }
#                     },
#                     "final_answer": {"type": "string"}
#                 },
#                 "required": ["steps", "final_answer"],
#                 "additionalProperties": False
#             },
#             "strict": True
#         }
#     }
#     )

#     return response.choices[0].message
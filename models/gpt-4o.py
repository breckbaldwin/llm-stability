from openai import AzureOpenAI
import os
import sys
from dotenv import load_dotenv

load_dotenv()

"""
Model: GPT4o hosted by Azure, see implementation for details on version
Authors: Breck Baldwin.
How to setup model to run: Model requires Azure Open AI authentication and endpoints as shown in code. If you need to setup an Azure Open AI account go to: https://azure.microsoft.com/en-us/products/ai-services/openai-service/. 
Tests: Run `pytest tests/test_models.py::test_gpt_4o` to verify that model functions. 


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

MODEL = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_ENDPOINT_GPT_4_0"],
                api_key=os.environ["OPENAI_GPT4_KEY"],
                api_version="2024-04-01-preview",
                azure_deployment="AppliedAI-gpt-4o",
            )
MODEL_NAME = "gpt-4o"

seen_before = {} #cache previous results for run to avoid variation

def run(prompt: list, config: dict) -> (str):
    """
    Runs a prompt and returns the content portion of the response as well as
    the actual run configuration. 
    Arguments:
       prompt: list of dicts, example: [{"role": "user", "content": question}]
       config: dict, {'temperature': float probability,
                      'seed': int,
                      'top_p_k': float probability,
                      'rewrite_inst': str
                      }
    Returns:
       str: payload
       run_info: {'prompt':str, 
                  'model_run': 'model_template', 
                  'config': {'temperature':probability, ...}
                 }
    """
    global seen_before
    cache_used = False
    temperature = config['temperature']
    seed = config['seed']
    top_p = config['top_p_k']
    rewrite_inst = config.get('rewrite_inst', None)
    if rewrite_inst is not None:
        original_content = prompt[0]['content']
        rewritten_content = None
        if original_content in seen_before:
            rewritten_content = seen_before[original_content]
            print(f"Cache hit '{original_content}' -> '{rewritten_content}'")
            cache_used = True
        else: 
            rewrite_prompt = [
                {"role": "user",
                 "content": f"{rewrite_inst}\n\n{original_content}"}
                ]
            rewrite_response = MODEL.chat.completions.create(
                    messages=rewrite_prompt,
                    model=MODEL_NAME,
                    temperature=temperature,
                    seed=seed,
                    top_p=top_p)
            rewritten_content = rewrite_response.choices[0].message.content
            print(f"Rewrote '{original_content}' to '{rewritten_content}'")
            seen_before[original_content] = rewritten_content
        prompt = [
                {"role": "user",
                 "content": rewritten_content}
                ]

    response = MODEL.chat.completions.create(
                    messages=prompt,
                    model=MODEL_NAME,
                    temperature=temperature,
                    seed=seed,
                    top_p=top_p
                )
    return (response.choices[0].message.content,
            {
            'prompt':prompt, 
            'model_name': MODEL_NAME, 
            'temperature': temperature,
            'seed': seed,
            'top_p_k': top_p,
            'rewrite_inst': rewrite_inst,
            'cache_used': cache_used
            })
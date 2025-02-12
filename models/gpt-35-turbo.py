from openai import AzureOpenAI
import os
import sys
from dotenv import load_dotenv

load_dotenv()

"""

Model: GPT 3.5 hosted by Azure, see implementation for details on version
Authors: Breck Baldwin.
How to setup model to run: Model requires Azure Open AI authentication and endpoints as shown in code. Easiest way to set variables is to have a `.env` file in your home directory for `dotenv` to find. DO NOT PUT `.env` in repo or 
you risk putting credentials in the repo. If you need to setup an Azure Open AI account go to: https://azure.microsoft.com/en-us/products/ai-services/openai-service/. 
Tests: Run `pytest tests/test_models.py::test_gpt_35_turbo` to verify that model functions. 


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
                azure_endpoint=os.environ["AZURE_ENDPOINT_GPT_3_5"],
                api_key=os.environ["OPENAI_API_KEY"],
                api_version="2023-07-01-preview",
                azure_deployment="AppliedAI-gpt-35-turbo",
            )
MODEL_NAME = "gpt-35-turbo"

seen_before = {} #cache previous results for run to avoid variation

def apply_rewrite(prompt, config):
    """
    Rewrites the content of a given prompt based on a configuration and cache.
    Args:
        prompt (list): A list containing a dictionary with the key 'content' representing the original prompt content.
        config (dict): A dictionary containing configuration parameters for the rewrite process. Expected keys are 'rewrite_inst' (instructions for rewriting), 'temperature', 'seed', and 'top_p_k'.
        A typical rewrite instruction is: "Please rewrite the below prompt to maximize your understanding of the instruction for processing."
    Returns:
        list: A list containing a dictionary with the key 'content' representing the rewritten prompt content.
        bool: A boolean flag indicating whether the cache was used.
    Side Effects:
        Updates the global 'seen_before' dictionary with the original content as the key and the rewritten content as the value.
        Prints messages indicating whether the cache was used or if the content was rewritten.
    """

    global seen_before
    cache_used = False
    original_content = prompt[0]['content']
    rewritten_content = None
    if original_content in seen_before:
        rewritten_content = seen_before[original_content]
        print(f"Cache hit '{original_content}' -> '{rewritten_content}'")
        cache_used = True
    else: 
        rewrite_prompt = [
            {"role": "user",
                "content": f"{config['rewrite_inst']}\n\n{original_content}"}
            ]
        rewrite_response = MODEL.chat.completions.create(
                messages=rewrite_prompt,
                model=MODEL_NAME,
                temperature=config['temperature'],
                seed=config['seed'],
                top_p=config['top_p_k'])
        rewritten_content = rewrite_response.choices[0].message.content
        print(f"Rewrote '{original_content}' to '{rewritten_content}'")
        seen_before[original_content] = rewritten_content
    prompt = [
            {"role": "user",
                "content": rewritten_content}
            ]
    return prompt, cache_used

def run(prompt: list, config: dict) -> (str):
    """
    Runs a prompt and returns the content portion of the response and 
    return run parameters.
    Arguments:
       prompt: list of dicts, example: [{"role": "user", "content": question}]
       config: dict, {'temperature': float probability,
                      'seed': int
                      'top_p_k': float probability,
                      'rewrite_inst': str
                      }
    Returns:
       str: payload
       dict: run configuration used--may have rewritten prompt or other mods
    """
    cache_used = False
    if config.get('rewrite_inst', None) is not None:
        prompt, cache_used = apply_rewrite(prompt, config)

    response = MODEL.chat.completions.create(
                    messages=prompt,
                    model=MODEL_NAME,
                    temperature=config['temperature'],
                    seed=config['seed'],
                    top_p=config['top_p_k']
                )
    return (response.choices[0].message.content,
            {
            'prompt':prompt, 
            'model_name': MODEL_NAME, 
            'temperature': config['temperature'],
            'seed': config['seed'],
            'top_p_k': config['top_p_k'],
            'rewrite_inst': config.get('rewrite_inst', None),
            'cache_used': cache_used
            })
import together
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from together import Together

MODEL = Together(api_key=os.environ["LLAMA3-70B-KEY"])

MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo"

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
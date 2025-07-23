import requests
import concurrent.futures

"""
Coded closer to the native llama.cpp server API just to keep it simple and direct
"""

MODEL_NAME = 'Llama-32-3B-Instruct-Q3_K_XL_non_det'

def send_job(prompt, i):
    res = requests.post("http://localhost:8080/completion", json={
        "prompt": prompt,
        "temperature": 0.0,
        "n_predict": 512,
        "top_k": 0,
        "top_p": 1.0,
        "stop": ["</s>"]
    })
    return i, res.json()["content"]



def run(prompt: list, config: dict) -> (str, dict):
    """
    Submits a prompt in a randomized position to test order effects, and returns
    the result and run metadata.

    Args:
        prompt: list of dicts, e.g., [{"role": "user", "content": question}]
        config: dict with keys:
            - 'temperature': float
            - 'seed': int
            - 'top_p_k': float
            - 'batch': bool
            - 'run': int (used for shuffling order)

    Returns:
        response: str, the output from the LLM for the target prompt
        run_info: dict, metadata about the actual test
    """
    primary_prompt = prompt[0]['content']
    filler_prompt = "Tell me about the simulation hypothesis"
    prompts = [primary_prompt, filler_prompt]
    
    answer_idx = 0
    if config['round'] % 2 == 1:
        prompts.reverse()
        answer_idx = 1
        print("running opposite order")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(send_job, p, i) for i, p in enumerate(prompts)]
        for f in concurrent.futures.as_completed(futures):
            i, output = f.result()
            if i == answer_idx:
                print(f"answer on {i}: {output}")
                return (
                    output,
                    {
                        'prompt': [{'role': 'user', 'content': prompts[i]}],
                        'model_name': MODEL_NAME,
                        'temperature': config['temperature'],
                        'seed': config['seed'],
                        'top_p_k': config['top_p_k']
                    }
                )

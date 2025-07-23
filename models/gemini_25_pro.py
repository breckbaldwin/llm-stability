import os
from dotenv import load_dotenv
from types import SimpleNamespace
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with your API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Use Gemini 2.5 Pro.
MODEL_NAME = "gemini-2.5-pro" # Or a specific version like "gemini-2.5-pro-001" if available and desired

seen_before = {} # Cache for 'rewrite_inst' - ensure its logic is deterministic.

# Placeholder for helper_functions.apply_rewrite
# This function must be deterministic itself for overall determinism.
def apply_rewrite(prompt, config, seen_before, model, model_name):
    """
    Placeholder for a helper function to apply rewrite instructions.
    For maximum determinism, ensure this function itself is deterministic.
    """
    if config.get('rewrite_inst'):
        print("Note: 'apply_rewrite' is a placeholder. For determinism, ensure its logic is consistent.")
    return prompt, False # Returns modified prompt and cache_used status


def run(prompt: list, config: dict) -> (str, dict):
    """
    Runs a prompt against the Gemini model, configured for maximum determinism,
    and returns the content of the response along with detailed run information.

    Arguments:
       prompt: list of dicts, e.g., [{"role": "user", "content": "Your question here"}]
       config: dict, containing configuration for the model run. Key parameters:
               'temperature': float (should be 0 for determinism)
               'top_p': float (should be 0 or very close to 0 for determinism)
               'rewrite_inst': str (optional, for pre-processing prompts)
               'prefix': str (optional, prepends to user content)
               'suffix': str (optional, appends to user content)
               'logprobs': bool (optional, whether to request log probabilities; limited support)
               'seed': int (This parameter will be noted in run_info but not passed to Gemini API directly
                           via GenerationConfig due to current API limitations in this SDK.)

    Returns:
       str: The content of the model's response.
       dict: A dictionary containing comprehensive information about the run,
             including the prompt, model details, configuration used, and log probabilities (if requested).
    """
    cache_used = False

    # Apply rewrite instruction if provided. Ensure apply_rewrite is deterministic.
    if config.get('rewrite_inst', None) is not None:
        prompt, cache_used = apply_rewrite(prompt, config, seen_before, genai, MODEL_NAME)

    # Apply prefix to the first user message if provided
    if config.get('prefix') is not None:
        if prompt and prompt[0].get('role') == 'user':
            prompt[0]['content'] = config['prefix'] + prompt[0]['content']
        else:
            prompt.insert(0, {'role': 'user', 'content': config['prefix']})

    # Apply suffix to the first user message if provided
    if config.get('suffix') is not None:
        if prompt and prompt[0].get('role') == 'user':
            prompt[0]['content'] = prompt[0]['content'] + config['suffix']
        else:
            prompt.append({'role': 'user', 'content': config['suffix']})

    # --- Deterministic Configuration for Gemini ---
    # 1. Temperature: Set to 0.0.
    # 2. Top-P: Set to 0.0.
    # 3. Candidate Count: Always request a single candidate.
    generation_config = {
        "temperature": config['temperature'], # Maximize determinism
        "top_p": config['top_p_k'],       # Maximize determinism
        "candidate_count": 1,
    }

    # Logprobs handling: Still the same limitation.
    logprobs_config = config.get('logprobs', False)
    if logprobs_config:
        print("Warning: Direct 'logprobs' like Together AI are not directly available in Gemini's chat completions API.")
        print("The 'logprobs' field in the returned run_info will be empty.")
        logprobs = []
    else:
        logprobs = []

    # Initialize the model
    model = genai.GenerativeModel(MODEL_NAME)

    # Convert prompt format for Gemini
    gemini_prompt = []
    for message in prompt:
        if message['role'] == 'user':
            gemini_prompt.append({'role': 'user', 'parts': [message['content']]})
        elif message['role'] == 'assistant':
            gemini_prompt.append({'role': 'model', 'parts': [message['content']]})

    try:
        response = model.generate_content(
            gemini_prompt,
            generation_config=generation_config,
        )
        response_content = response.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        response_content = f"Error: {e}"

    run_info = {
        'prompt': prompt,
        'model_name': MODEL_NAME,
        'temperature': generation_config['temperature'],
        'seed': config.get('seed', None), # Keep seed in run_info for tracking, even if not passed to API
        'top_p_k': generation_config['top_p'],
        'rewrite_inst': config.get('rewrite_inst', None),
        'cache_used': cache_used,
        'logprobs': logprobs
    }

    return response_content, run_info

if __name__ == '__main__':
    # Example Usage for Determinism:
    # Set your GEMINI_API_KEY in a .env file or as an environment variable

    deterministic_prompt = [{"role": "user", "content": "What is 2 + 2?"}]

    # Configuration for maximum determinism
    deterministic_config = {
        'temperature': 0.0,
        'seed': 12345, # This seed is for your tracking and external logic, not directly passed to Gemini's GenerationConfig via this SDK.
        'top_p': 0.0,
        'rewrite_inst': None,
        'prefix': "Strictly answer with only the numerical result: ",
        'logprobs': False
    }

    print("--- Running Deterministic Example 1 ---")
    content1, info1 = run(deterministic_prompt, deterministic_config)
    print(f"Response Content 1: {content1}")
    print(f"Run Info 1 (Temperature: {info1['temperature']}, Seed: {info1['seed']}, Top_P: {info1['top_p']})")

    print("\n--- Running Deterministic Example 2 (Same Input) ---")
    content2, info2 = run(deterministic_prompt, deterministic_config)
    print(f"Response Content 2: {content2}")
    print(f"Run Info 2 (Temperature: {info2['temperature']}, Seed: {info2['seed']}, Top_P: {info2['top_p']})")

    print("\n--- Running Deterministic Example 3 (Slightly Different Prompt) ---")
    different_prompt = [{"role": "user", "content": "What is the result of 5 times 3?"}]
    content3, info3 = run(different_prompt, deterministic_config)
    print(f"Response Content 3: {content3}")
    print(f"Run Info 3 (Temperature: {info3['temperature']}, Seed: {info3['seed']}, Top_P: {info3['top_p']})")
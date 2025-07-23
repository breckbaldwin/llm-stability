import os
from dotenv import load_dotenv
from types import SimpleNamespace
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with your API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Use Gemini 1.5 Pro.
MODEL_NAME = "gemini-1.5-pro" # Or a specific version like "gemini-2.5-pro-001" if available and desired

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

#Gemini response to my request for determinstic LLM
"""
G:-------
You're right to ask about Gemini 2.5! My previous response was based on the information available at the time, and with the rapid pace of LLM development, things can change quickly.

As of July 2025, you absolutely should consider using Gemini 2.5 models for determinism, especially Gemini 2.5 Pro.

Here's why:

    Gemini 2.5 Pro is now Generally Available (GA) and stable. This is a significant factor for production use and for seeking determinism. "Preview" or "experimental" models are more likely to have their behavior or underlying architecture change, which can impact reproducibility. As of June 17, 2025, both Gemini 2.5 Pro and Gemini 2.5 Flash became generally available.

Enhanced Reasoning and Capabilities: Gemini 2.5 Pro is explicitly designed as a "thinking model" with improved reasoning and coding capabilities. While this doesn't directly relate to determinism, it suggests a more robust and predictable internal process for complex tasks, which can indirectly contribute to more consistent outputs for well-defined prompts.

    Continued Focus on Reproducibility: While no LLM can guarantee 100% determinism due to the inherent complexity and potential for hardware/software variations, Google continues to offer and refine the seed parameter and encourages setting temperature and top_p to 0.0 for the most consistent results. These parameters apply to Gemini 2.5 models as well.

In summary:

    Use gemini-1.5-pro if you need to stick with a model that was stable a bit earlier in time and whose behavior you're already familiar with.

    For the best performance and current best effort at determinism, you should upgrade to gemini-2.5-pro. This is the latest stable Pro model and will offer the most advanced capabilities alongside the deterministic settings we discussed (temperature=0.0, top_p=0.0, and a fixed seed).

The wrapper I provided should work seamlessly with MODEL_NAME = "gemini-2.5-pro". Just update that line in your code.

Always check the Gemini API Release Notes for the absolute latest information on model availability, stability, and any new parameters that might affect determinism.

BB: -------
I get an error for the seed parameter, I am running 2.5


python gemini_25_pro.py        

--- Running Deterministic Example 1 ---

Error calling Gemini API: Unknown field for GenerationConfig: seed

Response Content 1: Error: Unknown field for GenerationConfig: seed

Run Info 1 (Temperature: 0.0, Seed: 12345, Top_P: 0.0) 

G:--------

You are absolutely correct to point that out! My apologies. There seems to be a mismatch between the documentation I was referencing (which indicated a seed parameter in GenerationConfig) and the actual implementation for google.generativeai client library with Gemini 2.5 Pro.

The error message "Unknown field for GenerationConfig: seed" clearly indicates that the seed parameter is not directly supported within the generation_config dictionary for the google.generativeai Python SDK when calling model.generate_content.

This is a common point of confusion, as the concept of "seed" for reproducibility often exists at the model level, but its exposure in specific client libraries or API versions can vary. Some APIs might expose it at a higher level (like a dedicated seed parameter in the create call directly), or it might only be available in certain environments (like Vertex AI's API, which has different client libraries and interfaces).

What does this mean for determinism with Gemini 2.5 Pro?

While the seed parameter is not exposed in generation_config via google.generativeai at this moment, you should still maximize determinism by doing the following:

    temperature = 0.0: This remains the most critical setting. It forces the model to select the most probable token at each step, significantly reducing randomness.

    top_p = 0.0: This reinforces the temperature=0.0 setting by ensuring only the top-probability token is considered.

    Fixed Model Version: Continue using MODEL_NAME = "gemini-2.5-pro" (or a specific version like "gemini-2.5-pro-001" if available and you want to lock to a very specific snapshot). Using a stable, named model helps ensure that the underlying model doesn't change unexpectedly.

Revised Wrapper (removing the problematic seed parameter from generation_config):

<code above>

"""
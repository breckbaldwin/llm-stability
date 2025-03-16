import streamlit as st
import sys
import os
import pandas as pd
from collections import defaultdict
import json
sys.path.append(os.path.join(os.getcwd(), "cells"))
import cells.cell_util
import re
import glob

ss = st.session_state


def run_ui(page, cell_i):
    st.markdown("""
### LLMs are not stable

- [https://arxiv.org/pdf/2408.04667](https://arxiv.org/pdf/2408.04667)
    + Hyper-parameters set to maximize determinism  
    + Stability/determinism is independent of hallucinations and errors
        - AI systems have always made mistakes--but have been stable on input
        - LLMs add a new dimension of instability on input
- Huge consequences for production engineering
    + With hyper-parameters set to maximize determinism we still see variation
    + No longer can reliably test on same inputs (unit tests)
    + 80/20 Rule broken
- Lab Week Fall 2024: Can we make LLMs stable? 
    1. Ask for one word answers (Breck) for multiple choice
        + Politely, 
        + .... then less politely
    2. Expand to open ended problems (Sarp)
        + Generate Python coding challenge questions
        + Evaluate by passing unit tests
    3. Understand what is going on under the hood (Tomasz)
        + Run locally open source models where we can control everything
        + Characterize data generating process
        + Low transparency from vendors, nerfed LLMs

| Parameter | Description | Value Type | Default Value |
|---------------------|-----------------------------------------------------------------------------------------------------|------------|---------------|
| num_ctx | Sets the size of the context window used to generate the next token. Depends on the model's limit. | int | 4096 |
| num_predict | Maximum number of tokens to predict during text generation. Use -1 for infinite, -2 to fill context.| int | -1 |
| temperature | Adjusts the model's creativity. Higher values lead to more creative responses. Range: 0.0-2.0. | float | 0.8 |
| repeat_penalty | Penalizes repetitions. Higher values increase the penalty. Range: 0.0-2.0. | float | 1.0 |
| repeat_last_n | How far back the model checks to prevent repetition. 0 = disabled, -1 = num_ctx. | int | 64 |
| top_k | Limits the likelihood of less probable responses. Higher values allow more diversity. Range: -1-100 | int | 40 |
| top_p | Works with top_k to manage diversity of responses. Higher values lead to more diversity. Range: 0.0-1.0. | float | 0.95 |
| tfs_z | Tail free sampling reduces the impact of less probable tokens. Higher values diminish this impact. | float | 1.0 |
| typical_p | Sets a minimum likelihood threshold for considering a token. Range: 0.0-1.0. | float | 1.0 |
| presence_penalty | Penalizes new tokens based on their presence so far. Range: 0.0-1.0. | float | 0.0 |
| frequency_penalty | Penalizes new tokens based on their frequency so far. Range: 0.0-1.0. | float | 0.0 |
| mirostat | Enables Mirostat sampling to control perplexity. 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0. | int | 0 |
| mirostat_tau | Balances between coherence and diversity of output. Lower values yield more coherence. Range: 0.0-10.0. | float | 5.0 |
| mirostat_eta | Influences response speed to feedback in text generation. Higher rates mean quicker adjustments. Range: 0.0-1.0. | float | 0.1 |
| num_keep | Number of tokens to keep unchanged at the beginning of generated text. | int | 0 |
| penalize_newline | Whether to penalize the generation of new lines. | bool | True |
| stop | Triggers the model to stop generating text when this pattern is encountered. List strings separated by ", ". | string Array | empty |
| seed | Sets the random number seed for generation. Specific numbers ensure reproducibility. -1 = random. | int | -1 |


## Conclusions

- We have NOT figured out how to make LLMs stable/deterministic
- Excellent progress on infrastructure
    + GUI's built
    + Connection to two new classes of experimentation--open ended, locally run
- Interest from our summer intern and their PhD advisor to continue work
- Foundational work that benefits entire field
    + Work is released publicly as open source/open data

 """)



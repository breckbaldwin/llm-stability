# LLM Stability Project:

Public repo: https://github.com/Comcast/llm-stability
Comcast repo: https://github.com/comcast-explainable-ai-lab-group/llm_stability


vNext.0 In development

## Supporting software and data

This repo contains source code and data used to conduct our experiments. There are some naming conventions around versions of software/papers.

- v1.0 is the original ArXive paper v1.0 Submitted July 26, 2024 ArXiv released August 6, 2024: https://arxiv.org/abs/2408.04667v1. There is very little support for this paper in the repo or write up.
- v2.0 (v2.0 https://arxiv.org/abs/2408.04667v2) is a superset of the v1.0 work and where effort has been made to release software and data in a reproducable way. However the release was a retrofit onto an existing project. 
- vNext.0 is the next version that attempts to be a stronger effort for reproducability and clarity in the code base. There are two major efforts:
    + Getting the v2.0 paper published somewhere. There are planned minor extensions to the v2.0 paper for resubmittal. 
    + Explore solutions to non-determinism in follow on work. 

Below is vNext.0 work. 

### Overview 

The project splits cleanly between running LLMs against benchmark tasks which output to
 .csv files and then evaluating those files. There are recreations of the v2 experiments in `experiments/v2_few_shot/`. There is a standardized evaluation script:

- `evaluate.py` that takes generated csv files from prior runs and produces a table output similar to Table 2 in https://arxiv.org/abs/2408.04667v2 in the file `evaluation_output.csv`. There is example .csv in the repo which can be evaluated by running:
```shell
python evaluate.py -d experiments/example/0001-01-01_00-00-00/
```

- `run_from_experiment_output.py` takes existing experiment output and will rerun from the configuration information in the output files. The only degree of freedom is the LLM used to run the task. If the hosted LLMs change then the experiments will not be reproducable. Note that the script is included for completeness sake and the data is hard-coded into the script. Example run is:

```
python run_from_experiment_output.py

```

- `run_experiment.py` will take an LLM, a task to evaluate and produce files for evaluation. Note that you will have to supply environment variables with endpoints and credentials for the models. This is covered in `models/README.md`. An example invocation is:

```

python run_experiment.py -m gpt-35-turbo -mc '{"temperature":0.0}' -t navigate -tc '{"prompt_type": "v2", "shots":"few"}' -n 5
```
- This will write 5 csv files to the directory `local_runs/<timestamp>/gpt-35-turbo-0.0-navigate-few-<run number>.csv` and `python run_experiment.py -h produces helpful output. 

The data from the experiment runs reported in v1.0 (few-shot) and the additional runs for various alternate configurations (few-shot, 0-shot and fine-tuned) in v2.0 are in `release_data.tgz` which can be uncompressed with the command line utility `tar -xvzf release_data.tgz` on linux/osx or which results in a folder `release_data`. GUI window managers also typcially can extract the compressed data as well by double clicking on the file. It has sub folders
* `release_data/few_shot` which are the results from the few shot runs repeated 5 times. This contains the preliminary data that setup the paper. 
* `release_data/20_run` which is the 20 run example used to help characterize the shape of score variations.
* `release_data/0-shot` is a collection of 5 run evaluations with no training examples provided.
* `release_data/fine-tuned-few-shot` contain the results of fine tuning several models on some of the tasks. 

The above data represents 520 runs ranging from 100 to 250 questions each and as such there remains a great deal of potential analysis over the experiments we ran but did not conduct. We release this data in the hopes of others taking advantage of a rather expensive and time consuming effort. 


## Setup

Our LLMs were hosted on a variety of services that are specific to our company. But it should be easy to point the LLMs at the appropriate hosted endpoints. Look at 
`


In addition there is a `requirements.txt` file that details the modules needed to run the software. Typically one creates a Python virtual environment and then run `pip install -r requirements.txt` for installation. 

## Explanations for the files

The files below are offered more in the hopes of clarity about what was done exactly for the purposes of reproducability than serving as a foundation for related work. The processes are just not that complex and our implementation not intended to live beyond the needs of the experiments but we have tried to make the code and steps we took as clear as possible. 

- `main.py`: This is the main file to run the LLMs. It has 4 named parameters:
  - `task`: which is to specify a task which can be any of the followings: mmlu_professional_accounting,  bbh_geometric_shapes, bbh_logical_deduction_three_objects, bbh_navigate, bbh_ruin_names, mmlu_college_mathematics, mmlu_high_school_european_history
  - `model`: the name of the model such as gpt-3.5-turbo, gpt-4o, or llama-3-70b
  - `num_few_shot`: number of few shot examples being used, they are set in the paper as follows for the few shot examples : bbh:3, mmlu:5. There are also 0-shot runs. 
  - `experiment_run`: this is to keep track of different runs, 5 for most tasks with some runs at 20 to characterize the shape of the accuracy distribution.
  - Example call: `python main.py  --model gpt-3.5-turbo --task mmlu_professional_accounting --experiment_run 0 --num_fewshot 4`. Running this will create a file `gpt-3.5-turbo_mmlu_professional_accounting_0.csv` that is a sibling to `main.py`. Note that the `num_fewshot` parameter was not included in the output filename convention and was added by hand later. We suggest collecting runs into an appropriately named folder as done in our `release_data` folder.
- `helper_functions.py`: Helper functions here that are being used in other files.
- `postprocess_responses.py`: We encountered parsing complexities in extracting answers.  This file after a run is done to get rid of some of the parsing issues but sometimes manual checking is inevitable. 
- `calculate_statistics.py`: This calculates the reported metrics after all parsings are done. An example call is: `python calculate_statistics.py release_data/few_shot fewshot_scores` which creates the indicated folder if needed and writes the files:
    + `fewshot_scores/detail_accuracy.csv`: Parsed outputs for each model/task/run
    + `fewshot_scores/low_med_high.csv` Accuracy, low, median, high
    + `fewshot_scores/TARa.csv` Total agreement rate answer results
    + `fewshot_scores/TARr.csv` Total agreement rate raw results

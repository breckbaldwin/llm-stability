import run_experiment
import evaluate
import importlib
from datetime import datetime, date


data_df = evaluate.load_runs("local_runs/2024-12-14_18-33-44/")
date = datetime.now()
datetime_string = date.strftime("%Y-%m-%d_%H-%M-%S")

num_rubrics = 3
num_runs = 2
configs = evaluate.get_experiment_configs(data_df)
for model, model_config, task, task_config in configs:
    llm = importlib.import_module(f"models.{model}")
    exp_df = data_df[(data_df['model'] == model)
                        & (data_df['model_config'] == model_config)
                        & (data_df['task'] == task)
                        & (data_df['task_config'] == task_config)
                        & (data_df['run'] == 0)]
    run_experiment.run_model(llm, 
                            out_dir="tmp", 
                            outfile_root="testing", 
                            num_runs=num_runs,
                            model_config=model_config,
                            test_rubrics=exp_df['rubric'][:num_rubrics],
                            model_name=model,
                            task_name=task,
                            task_config=task_config,
                            date=datetime_string,
                            context=None)





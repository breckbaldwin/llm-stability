import streamlit as st
import pandas as pd
import os


ss = st.session_state


def load_data():
    if "exps_df" not in ss:
        exp_dfs = []
        for filename in os.listdir("../data/5_run"):
            if not filename.endswith(".csv"):
                continue
            print(f"loading {filename}")
            try:
                df = pd.read_csv(os.path.join("../data/5_run/", filename))
            except Exception as e:
                print(f"Got exception {e}, Failed to load {file}, keeping going")
                st.warning(f"Got exception {e}, Failed to load {file}, keeping going")
            df["filename"] = filename
            (model_task, run) = filename.rsplit("_", 1)
            (model, task) = model_task.split("_", 1)
            df["model"] = model
            df["task"] = task
            df["run"] = run.split(".")[0]
            df["question_num"] = df.index
            if "new_extracted_pred" not in df or df["new_extracted_pred"] is None:
                df["answer"] = df["pred"]
            else:
                df["answer"] = df["new_extracted_pred"]
            exp_dfs.append(df)

        ss.exps_df = pd.concat(exp_dfs, ignore_index=True)
        ss.exps_df.sort_values(by="filename", inplace=True)
        ss.sheet_index = 0

    if "filtered_exps_df" not in ss:
        ss.filtered_exps_df = ss.exps_df

    if "tasks" not in ss:
        task_dfs = []
        for filename in os.listdir("../data/tasks"):
            if not filename.endswith(".csv") or "no_shot" in filename:
                continue
            try:
                df = pd.read_csv(os.path.join("../data/tasks/", filename))
            except Exception as e:
                print(f"Got exception {e}, Failed to load {filename}, keeping going")
                st.warning(
                    f"Got exception {e}, Failed to load {filename}, keeping going"
                )
            df["filename"] = filename
            df["task"] = filename.replace("_raw_inputs.csv", "")
            df["question_num"] = df.index
            task_dfs.append(df)
        ss.tasks = pd.concat(task_dfs, ignore_index=True)

    if "tasks_no_shots" not in ss:
        task_dfs = []
        for filename in os.listdir("../data/tasks"):
            if not filename.endswith(".csv") or "no_shot" not in filename:
                continue
            try:
                df = pd.read_csv(os.path.join("../data/tasks/", filename))
            except Exception as e:
                print(f"Got exception {e}, Failed to load {filename}, keeping going")
                st.warning(
                    f"Got exception {e}, Failed to load {filename}, keeping going"
                )
            df["filename"] = filename
            df["task"] = filename.replace("_raw_inputs_no_shot.csv", "")
            df["question_num"] = df.index
            task_dfs.append(df)
        ss.tasks_no_shots = pd.concat(task_dfs, ignore_index=True)


def render_mismatches(runs_df):
    done = False
    start = 0
    increment = 80
    row_0 = runs_df.iloc[0]
    st.markdown("-----------")
    question = ss.tasks[
        (ss.tasks["task"] == row_0["task"])
        & (ss.tasks["question_num"] == row_0["question_num"])
    ].iloc[0]["raw_input"]
    st.expander("Question with shots").markdown(question)

    question_no_shot = ss.tasks_no_shots[
        (ss.tasks_no_shots["task"] == row_0["task"])
        & (ss.tasks_no_shots["question_num"] == row_0["question_num"])
    ].iloc[0]["raw_input"]
    st.expander("Question no shots").markdown(question_no_shot)
    st.expander("Run 0 Response").markdown(row_0["raw_response"])
    st.markdown(f"{row_0['model']}, {row_0['task']}, question {row_0['question_num']}")

    while not done:
        end = start + increment
        for i in range(1, 5):
            row = runs_df.iloc[i]
            if row_0["raw_response"][start:end] != row["raw_response"][start:end]:

                correct = row["gt"] == row["answer"] and row["gt"] == row_0["answer"]
                result = "Correct" if correct else "Wrong"

                st.markdown(
                    f"{result} -- Truth: {row['gt']}, Run 0: {row_0['answer']}, Run {i}: {row['answer']}, "
                )
                st.markdown(f"0: {row_0['raw_response'][start:end]}")
                st.markdown(f"{row['run']}: {row['raw_response'][start:end]}")
                st.expander(f"Run {i} Response").markdown(row["raw_response"])
                done = True
        start = end


def filter():
    if ss.task_sb == "All":
        ss.filtered_exps_df = ss.exps_df
    else:
        ss.filtered_exps_df = ss.exps_df[ss.exps_df["task"] == ss.task_sb]
    if ss.model_sb != "All":
        ss.filtered_exps_df = ss.filtered_exps_df[
            ss.filtered_exps_df["model"] == ss.model_sb
        ]
    if ss.question_sb != "All":
        ss.filtered_exps_df = ss.filtered_exps_df[
            ss.filtered_exps_df["question_num"] == ss.question_sb
        ]
    ss.sheet_index = 0


# Rendering starts here

load_data()

st.expander("Show all tasks").dataframe(ss.tasks)

st.expander("Show tasks without shots").dataframe(ss.tasks_no_shots)

filter_cols = st.columns(4)

filter_cols[0].selectbox(
    "Filter on task",
    options=["All"] + list(ss.tasks["task"].unique()),
    # on_change=filter,
    key="task_sb",
)

filter_cols[1].selectbox(
    "Filter on LLM",
    options=["All"] + list(ss.filtered_exps_df["model"].unique()),
    # on_change=filter,
    key="model_sb",
)

filter_cols[2].selectbox(
    "Select question",
    options=["All"] + sorted(list(ss.filtered_exps_df["question_num"].unique())),
    # on_change=filter,
    key="question_sb",
)

if filter_cols[3].button("Apply Filters"):
    filter()

st.expander(f"Show LLM data: {len(ss.filtered_exps_df.index)} rows").dataframe(
    ss.filtered_exps_df
)

if st.button("Show Next Mismatch"):
    ss.sheet_index += 1

go = True
while ss.sheet_index < len(ss.filtered_exps_df.index) and go:
    # breakpoint()
    row_next = ss.filtered_exps_df.iloc[ss.sheet_index]
    runs_df = ss.filtered_exps_df[
        (ss.filtered_exps_df["model"] == row_next["model"])
        & (ss.filtered_exps_df["task"] == row_next["task"])
        & (ss.filtered_exps_df["question_num"] == row_next["question_num"])
    ]
    if len(runs_df["raw_response"].unique()) > 1:  # we have distinct responses
        render_mismatches(runs_df)
        go = False
    else:
        print(f"Incrementing ss.sheet_index {ss.sheet_index}")
        ss.sheet_index += 1

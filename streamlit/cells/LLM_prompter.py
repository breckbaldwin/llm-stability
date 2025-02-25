import streamlit as st
import os
import requests
import json

from collections import defaultdict
import logging
import sys
import glob
from dotenv import load_dotenv
from datetime import datetime
import plotnine as p9
import statistics
import pandas as pd
import re
from utils.constants import CID, CUST, AGNT, SYST, TIME, SEQ, SID, ANNOT

ss = st.session_state

PROMPT = "prompt"
PROMPTS = "prompts"
INCLUDE = "include"
PROMPT_DATA = "prompt_data"
GOAL = "beat"
TYPE = "type"
LABEL = "beat label"
POSSIBLE_VALUES = "possible values"
TOTAL = "total"
CORRECT = "correct"
SETUP = "setup"

TEXT = "text"
FILTER = "filter"
REMOVE = "remove"
INPUT = "input"
OUTPUT = "output"


# Add LLM application to single beat as possible
# Eval LLM against human annotated data--need count filter
# Run naive prompt LLM
# Run optimized prompt LLM
# Cache runs?
# Click on add fails
# Delete beat reader resets the 'choose beats' UI
# Get calibrate working
# Suggest format improvements to segment payload via LLM


# Apply standard ML models to task as options with eval

# Beat label can be excluded from prompt
# Add cell that hill climbs the prompt

# Stop dump of json on save
# Edit button to expand cell
# Lightweight view into prompt
# Add All to beats
# Per cell edit
# On save re-order based on first beat for prompts, stop spreading out beat parts
# Sort out adding process, sort of awkward
# Add Label, prompt, reader as single command
# Make clear via font color active beats
# Add warning if no system or shared component when running
# Encode Pass/Fail into 6 binary classifiers with their own confidence
# Do 6 binary hierarchical to address sparse data
# Need to think about "I don't know" cases. Turns whole system into 3 value system
# x print mean value on graph
# Consider leveraging stochastic nature of LLM classifier to get free concentration by re-running multiple examples past classifier. Have to measure variance of running same chat by same prompt but with some stochastic elements from the LLM if temperature is high enough. A prompt that performs the same across higher temperatures will merit concentration. Alternatively we can say that if the prompt varies at all we should discount the concentration available from the chat x prompt pairing.
# Also, run prompts with other LLMs to increase concentration
# The prompts are a very interesting artifact with power over that of a standard NN supervised approach
# Split eval into false positive/false negatives from single correct/incorrect
# Impose beats as integers with an ordering
#
# P(phi|y) = P(y|phi) * P(phi)/p(y)
# P(TN|n1) = P(n1|TN) * P(TN)/p(n1)
# P(TP|y1) = P(y1|TP) * P(TP)/p(y1)

# Until I have seen a TN I can't trust my classifier
# At Bloomberg we ran classifiers to collect low prevalence data without
# regard for recall. We established signal and then thresholded to regulate precision. From that I am thinking that a classifier has no value until it has been calibrated on both sides for low prevalence phenomenon. Bayes' rule doesn't help since the difference between an always y1 and a 999/1000 y1, 1/1000 n1 is minimal.

# How do I make an uncertainty estimation sensitive to the above?
# Simulation is statistically required to assess system performance at low
# prevalence levels. Then the science is measuring the difference between the DGP sim and the world.
#

try:
    import cells.cell_util as cell
except ModuleNotFoundError as e:
    import cell_util as cell

if "nothing_to_save" not in ss:
    ss.nothing_to_save = True

if "LLM_prompter_exp" not in ss:
    ss.LLM_prompter_exp = False

BLANK_PROMPT = {GOAL: None, INCLUDE: True, PROMPT: ""}

BLANK_READER = {
    GOAL: None,
    TYPE: "reader",
    INCLUDE: True,
    POSSIBLE_VALUES: ["", ""],
    CORRECT: 0,
    TOTAL: 0,
}

BLANK_HILL_CLIMBER = {
    GOAL: None,
    TYPE: "hill_climber",
    INCLUDE: True,
    LABEL: False,
    PROMPT: "",
}


# Modes:
# Refine against single chat
# Lishing approach of refine against existing gold standard annotations
# Refine single goal (classification/extraction/summarization) against single chat at a time.
# add library of classifiers
#

INITS = ["", f"<{TEXT}> <{FILTER}> <{REMOVE}> <{INPUT}=?> <{OUTPUT}=?>"]


def update_dict(dict, key, widget_name):
    dict[key] = ss[widget_name]


if "last_turn" not in ss:
    ss.last_turn = 0

def get_beats_to_use(i_cell):
    if (ss.enable_edit_cb 
        and len(ss[f"beats_{i_cell}_ms"]) == 1 
        and ss[f"beats_{i_cell}_ms"][0] != 'All'):
            return ss[f"beats_{i_cell}_ms"]
    else:
        return sorted(
            list(set(entry[GOAL] for entry in ss.setup))
        )

def update_prompt(page, i_cell, i, widget):
    ss.config[page][i_cell]["setup"][i][PROMPT] = ss[widget]
    ss.nothing_to_save = False


def update_include(page, i_cell, i, widget):
    ss.config[page][i_cell]["setup"][i][INCLUDE] = ss[widget]
    ss.nothing_to_save = False


def update_label(page, i_cell, i, widget):
    ss.config[page][i_cell]["setup"][i][LABEL] = ss[widget]
    ss.nothing_to_save = False


def chat_to_str(chat_df):
    row_strs = []
    for i, row in chat_df.iterrows():
        cols_vals = [f"{row[col]}" for col in chat_df.columns]
        row_strs.append(". ".join(cols_vals))
    return "\n".join(row_strs)


def insert_chat(prompt_id, chat_df):

    ss.p_d[PROMPT_DATA][prompt_id][PROMPT] = chat_to_str(chat_df)
    ss.nothing_to_save = False


# def read_ui(page, i_cell): 
#     for i, entry in enumerate(ss.config[page][i_cell]["setup"]):
#         #if entry["beat"] not in ss[f"beats_{i_cell}_ms"]:
#         #    continue
#         ss.config[page][i_cell]["setup"][i][INCLUDE] = ss[
#             f"include_{i_cell}_{i}_cb"
#         ]
#         ss.config[page][i_cell]["setup"][i][LABEL] = ss[
#             f"label_{i_cell}_{i}_cb"
#         ]
#         if entry.get("type", "") == "reader":
#             value_remains = True
#             value_j = 0
#             while value_remains:
#                 if f"pv_{i_cell}_{i}_{value_j}" in ss:
#                     ss.config[page][i_cell]["setup"][i][POSSIBLE_VALUES][
#                         value_j
#                     ] = ss[f"pv_{i_cell}_{i}_{value_j}"]
#                     value_j += 1
#                 else:
#                     value_remains = False
#         else:
#             ss.config[page][i_cell]["setup"][i][PROMPT] = ss[
#                 f"prompt_{i_cell}_{i}_ta"
#             ]
#     # st.json(ss.config[page][i_cell]["setup"])


def update_ss_from_widget(var, widget):
    ss[var] = ss[widget]


def save_config(page, i_cell):
    #read_ui(page, i_cell)
    cell.save_config(ss)
    st.info(f"file saved : configs/{ss.config_file}")


def add_cell(page, i_cell, i, widget, last):
    val = ss[widget]
    if ss.config[page][i_cell][SETUP] == []:
        goal_name = "1"
        ss.goal_setup = goal_name
    elif last:
        goal_name = ss.config[page][i_cell][SETUP][i][GOAL]
    else:
        goal_name = ss.config[page][i_cell][SETUP][i][GOAL]
    if val == "Reader":
        cell = BLANK_READER.copy()
    elif val == "Hill Climber":
        cell = BLANK_HILL_CLIMBER.copy()
    else:
        cell = BLANK_PROMPT.copy()
    cell["beat"] = goal_name
    if last:
        ss.config[page][i_cell][SETUP].append(cell)
    else:
        ss.config[page][i_cell][SETUP].insert(i, cell)
    ss.nothing_to_save = False


def change_beat(page, i_cell, i, widget):
    ss.config[page][i_cell]["setup"][i][GOAL] = ss[widget]
    ss.nothing_to_save = False


def change_reader(page, i_cell, i, j):
    ss.config[page][i_cell]["setup"][i]['possible values'][j] = \
        ss[f"pv_{i_cell}_{i}_{j}"]
    ss.nothing_to_save = False


def edit_prompts_ui(ss, page, i_cell):
    Edit_exp = st.container()
    # if not ss.enable_edit:
    # return
    Edit_cols = Edit_exp.columns(3)
    Edit_cols[1].button(
        "Save Configuration",
        on_click=save_config,
        args=[page, i_cell],
        key=f"{i_cell}_save_LLM_prompt",
        disabled=ss.nothing_to_save,
    )
    beats = sorted(
        list(set(entry[GOAL] for entry in ss.config[page][i_cell][SETUP]))
    )
    Edit_cols[2].multiselect(
        "Beats",
        options=["All"] + beats,
        default=["All"],
        key=f"beats_{i_cell}_ms",
    )
    if ss[f"beats_{i_cell}_ms"] == []:
        Edit_exp.markdown("No beats selected")

    last_i = 0
    beats_to_use = get_beats_to_use(i_cell)
    for i, entry in enumerate(ss.config[page][i_cell]["setup"]):
        if entry["beat"] not in beats_to_use:
            continue
        Cols = Edit_exp.columns([1, 4])
        if POSSIBLE_VALUES in entry:
            Cols[1].markdown("Match in LLM output as labels")
            for j, possible_val in enumerate(entry[POSSIBLE_VALUES]):
                Cols[1].text_input(
                    "Label",
                    value=possible_val,
                    on_change=change_reader,
                    args=[page, i_cell, i, j],
                    label_visibility="collapsed",
                    key=f"pv_{i_cell}_{i}_{j}",
                )
        Cols[0].checkbox(
            INCLUDE,
            value=entry[INCLUDE],
            on_change=update_include,
            args=[page, i_cell, i, f"include_{i_cell}_{i}_cb"],
            key=f"include_{i_cell}_{i}_cb",
        )
        # Cols[0].checkbox(
        #     "Beat Label",
        #     value=entry[LABEL],
        #     on_change=update_label,
        #     args=[page, i_cell, i, f"label_{i_cell}_{i}_cb"],
        #     key=f"label_{i_cell}_{i}_cb",
        # )
        Cols[0].text_input(
            "Beat",
            value=entry["beat"],
            disabled=False,
            on_change=change_beat,
            args=[page, i_cell, i, f"beat_{i_cell}_{i}_ti"],
            label_visibility="collapsed",
            key=f"beat_{i_cell}_{i}_ti",
        )
        Cols[0].selectbox(
            "Add Above",
            options=["", "Blank", "Reader", "Hill Climber"],
            on_change=add_cell,
            args=[
                page,
                i_cell,
                i,
                f"add_cell_{i_cell}_{i}_{ss.odd_even}_sb",
                False,
            ],
            key=f"add_cell_{i_cell}_{i}_{ss.odd_even}_sb",
        )
        if Cols[0].button(
            "Del",
            key=f"del_prompt_{i_cell}_{i}",
        ):
            del ss.config[page][i_cell]["setup"][i]
            ss.nothing_to_save = False
            st.rerun()

        if PROMPT in entry:
            if len(entry[PROMPT]) > 60:
                Cols[1].text_area(
                    "Enter prompt here:",
                    value=entry[PROMPT],
                    on_change=update_prompt,
                    args=[page, i_cell, i, f"prompt_{i_cell}_{i}_ta"],
                    label_visibility="collapsed",
                    key=f"prompt_{i_cell}_{i}_ta",
                )
            else:
                Cols[1].text_input(
                    "Enter prompt here:",
                    value=entry[PROMPT],
                    on_change=update_prompt,
                    args=[page, i_cell, i, f"prompt_{i_cell}_{i}_ta"],
                    label_visibility="collapsed",
                    key=f"prompt_{i_cell}_{i}_ta",
                )
        last_i = i
        Cols[0].markdown("---------")

    Edit_exp.selectbox(
        "Add cell to end",
        options=["", "Blank", "Reader", "Selected Chat"],
        on_change=add_cell,
        args=[
            page,
            i_cell,
            last_i,
            f"add_cell_{i_cell}_{len(ss.config[page][i_cell]['setup'])}_sb",
            True,
        ],
        key=f"add_cell_{i_cell}_{len(ss.config[page][i_cell]['setup'])}_sb",
    )

    Edit_exp.write("-------------")
    #read_ui(page, i_cell)


# Start:


def stage_segment(chat_id, seg_id):
    # ss.chat_segments contains
    chat_df = ss.filtered_data[ss.filtered_data[CID] == chat_id]
    ss.chat_segments = cell.get_runnable_chat(ss.data_map, chat_df)

    # cell.get_segments_to_annotate(
    #    [ss.current_run_chat], ss.chat_prompts
    # )
    ss.current_seg_index = seg_id


def select_chat():
    ss.current_run_chat = ss[f"pick_a_chat_{ss.odd_even}_sb"]
    stage_segment(ss.current_run_chat, 0)


def select_segment():
    ss.current_seg_index = ss[f"segment_{ss.current_run_chat}_sb"]
    ss.run_results = None


def div_0_str(num, denom):
    try:
        expected_P = f"{num / denom:.1%}"
    except ZeroDivisionError as e:
        expected_P = "NA"
    return expected_P


def process_new_data(page, i_cell, beats_to_use, num_chats, num_runs):
    pooled_score = {"good_bot": 0, "total": 0}
    non_pooled_score = defaultdict(lambda: defaultdict(int))
    beat_to_class_to_chat_id = defaultdict(lambda: defaultdict(list))
    chat_id_to_beat_to_class = defaultdict(lambda: defaultdict(list))
    for i in range(num_chats):
        for j in range(num_runs):
            ss.current_run_chat = ss.chats[i]
            ss.current_seg_index = 0
            # ss.response_json = None
            ss.run = True
            stage_segment(ss.current_run_chat, ss.current_seg_index)
            response = run_segment(
                seg_index=ss.current_seg_index,
                segments=ss.chat_segments,
                page=page,
                i_cell=i_cell,
                beats_to_use=beats_to_use,
                temp=ss.temperature,
                top_p=ss.top_p,
                max_tokens=ss.max_tokens,
            )
            response_text = response.choices[0].message.content
            print(f"Extracted LLM response new data: {response_text}")
            results = cell.extract_LLM_results(
                response_text,
                ss.config[page][i_cell]["setup"],
                beats_to_use,
            )
            print(f"Results extracted: {results}")
            for beat, classification in results.items():
                beat_to_class_to_chat_id[beat][classification].append(
                    ss.current_run_chat
                )
                chat_id_to_beat_to_class[ss.current_run_chat][beat].append(
                    classification
                )
                if classification is not None and re.match(
                    r"[Y|y]es", classification
                ):
                    pooled_score["good_bot"] += 1
                    non_pooled_score[beat]["good_bot"] += 1
                pooled_score["total"] += 1
                non_pooled_score[beat]["total"] += 1
    return (
        pooled_score,
        non_pooled_score,
        beat_to_class_to_chat_id,
        chat_id_to_beat_to_class,
    )


def calibrate(
    page,
    i_cell,
    beats_to_use,
    temp,
    top_p,
    max_tokens,
    count
):
    beat_scores = defaultdict(lambda: defaultdict(int))
    accum_d = {"Beat": [], "Correct": [], "Total": []}
    if 'annot_coll_LLM' not in ss:
        ss.annot_coll_LLM = ss.annot_db[ss.collection_name]
    if 'eval_ids' not in ss:
        ss.eval_ids = list(ss.annot_coll_LLM.distinct('_id'))
    eval_ids = ss.eval_ids
    for id in eval_ids[0:count]:
        print(f"Running {id}")
        annot_found = ss.annot_coll_LLM.find_one({'_id': id})
        annotations = json.loads(annot_found[ANNOT])
        prompt, rubric = cell.assemble_prompt(
            ss.config[page][i_cell][SETUP], annotations["Data"], beats_to_use
        )
        result_json = cell.run_prompt(prompt, temperature=temp)
        response_text = result_json.choices[0].message.content
        print(f"Extracted LLM response calibration: {response_text}")
        results = cell.extract_LLM_results(
            response_text, ss.config[page][i_cell]["setup"], beats_to_use
        )
        print(f"Results extracted: {results}")
        for beat_name, value in results.items():
            if beat_name not in annotations: #no data, update pooled
                beat_scores[beat_name]["correct"] = 0
                beat_scores[beat_name]["total"] = 0
            # results_accum[beat][value] += 1
            if (
                value is not None
                # and re.match(r"[Yy]es", value)
                and beat_name in annotations
            ):
                if annotations[beat_name] == value:
                    beat_scores[beat_name]["correct"] += 1
                    beat_scores["pooled"]["correct"] += 1
                beat_scores[beat_name]["total"] += 1
                beat_scores["pooled"]["total"] += 1
    for beat_name, value in beat_scores.items():
        if beat_name == "pooled":
            continue
        accum_d["Beat"].append(beat_name)
        accum_d["Correct"].append(beat_scores[beat_name]["correct"])
        accum_d["Total"].append(beat_scores[beat_name]["total"])
    return beat_scores, accum_d


def no_labels(values):
    return [""] * len(values)


def show_pooled(Context, accum_d_l, pooled_score):
    colors = {"calibration": "grey", "predicted": "#f409d5"}
    accum_d = accum_d_l[0]
    binomial_draws_df = cell.binomial_pooled(
        N=len(accum_d["Beat"]), K=accum_d["Total"], y=accum_d["Correct"]
    )
    mean = statistics.mean(binomial_draws_df["phi"])
    try:
        pooled_result = pooled_score["good_bot"] / pooled_score["total"]
    except ZeroDivisionError as e:
        pooled_result = 0
    binomial_draws_df["phi_hat"] = binomial_draws_df.apply(
        lambda row: row.phi * pooled_result, axis=1
    )
    plot = (
        p9.ggplot(binomial_draws_df)
        + p9.geom_histogram(
            p9.aes(x="phi", y=p9.after_stat("count"), fill='"grey"'),
            alpha=0.4,
            binwidth=0.05,
        )
        # + p9.geom_histogram(p9.aes(x="phi", y=p9.after_stat("count")), fill="grey", alpha=0.4)
        + p9.geom_histogram(
            p9.aes(x="phi_hat", y=p9.after_stat("count"), fill='"#f409d5"'),
            alpha=0.4,
            binwidth=0.05,
        )
        # + p9.geom_vline(p9.aes(xintercept= mean))
        + p9.scale_x_continuous(
            name="phi, phi*measurement",
            labels=lambda l: ["%d%%" % (v * 100) for v in l],
            limits=(0, 1),
        )
        + p9.scale_y_continuous(name="", breaks=[], labels=no_labels)
        # + p9.annotate("text", x=mean, y=-20, label=f"{mean:.1%}", angle=0, size=20, color="#f409d5")
        + p9.scale_fill_identity(
            guide="legend",
            name="Uncertainty Distribution",
            breaks=["grey", "#f409d5"],
            labels=["Calibration", "Calibration * Measurement"],
        )
        # + p9.scale_fill_manual(values=colors)
        + p9.labs(
            title="Measurement Uncertainty via Calibration",
            subtitle="Histogram of draws for classifier and impact on measurement",
        )
        + p9.theme_xkcd()
        # + p9.theme(axis_text_x=None)
    )

    Context.expander("Explanation").markdown(
        """
Calibration treats false positives the same as false negatives which loses signal.
    """
    )
    Results_cols = Context.columns([2, 1])
    Results_cols[1].pyplot(p9.ggplot.draw(plot))

    correct_ratio = div_0_str(pooled_score["good_bot"], pooled_score["total"])
    for i in range(len(accum_d_l)):
        pooled_correct = sum(accum_d_l[i]["Correct"])
        pooled_total = sum(accum_d_l[i]["Total"])
        expected_P = div_0_str(pooled_correct, pooled_total)
        Results_cols[0].markdown(
            f"Mean calibration {i} : {expected_P} = {pooled_correct} / {pooled_total}"
        )
    Results_cols[0].markdown(
        f"Measurement: {correct_ratio} = {pooled_score['good_bot']} / {pooled_score['total']}"
    )
    try:
        AMA_pooled = (
            pooled_correct
            / pooled_total
            * pooled_score["good_bot"]
            / pooled_score["total"]
        )
    except ZeroDivisionError as e:
        AMA_pooled = 0.0
    Results_cols[0].markdown(
        f"Measured and calibrated performance: {AMA_pooled:.1%} = {expected_P} * {correct_ratio}"
    )


def show_non_pooled(
    Context,
    calib_accum_d_l,
    non_pooled_score,
    beat_to_class_to_chat_id,
    chat_id_to_beat_to_class,
):

    # Context.markdown("## Not Pooled")
    calib_accum_d = calib_accum_d_l[0]
    not_pooled_draws_df = cell.binomial_not_pooled(
        N=len(calib_accum_d["Beat"]),
        K=calib_accum_d["Total"],
        y=calib_accum_d["Correct"],
    )
    for i, (beat, counts) in enumerate(non_pooled_score.items()):
        try:
            value = counts["good_bot"] / counts["total"]
        except ZeroDivisionError as e:
            value = 0
        not_pooled_draws_df[f"theta_hat.{i + 1}"] = not_pooled_draws_df.apply(
            lambda row: row[f"theta.{i + 1}"] * value, axis=1
        )
    for i in range(0, len(calib_accum_d["Beat"])):
        Context.markdown("-------------")
        beat = calib_accum_d["Beat"][i]
        mean = statistics.mean(not_pooled_draws_df[f"theta.{i + 1}"])
        plot = (
            p9.ggplot(not_pooled_draws_df)
            + p9.geom_histogram(
                p9.aes(x=f"theta.{i + 1}", y=p9.after_stat("count")),
                fill="grey",
                alpha=0.4,
            )
            + p9.geom_histogram(
                p9.aes(x=f"theta_hat.{i + 1}", y=p9.after_stat("count")),
                fill="#f409d5",
                alpha=0.4,
            )
            + p9.geom_vline(p9.aes(xintercept=mean))
            + p9.scale_x_continuous(limits=(0, 1))
            + p9.annotate(
                "text",
                x=mean,
                y=-20,
                label=f"{mean:.1%}",
                angle=0,
                size=40,
                color="#f409d5",
            )
        )
        Results_cols = Context.columns([1, 3])
        if ss.dist_or_chat_r == "distributions":
            Results_cols = Context.columns([3, 1])
            Results_cols[1].pyplot(p9.ggplot.draw(plot))
        else:
            chats_d = {"id": [], "class": [], "user": [], "agent": []}
            # Results_cols[1].json(beat_to_class_to_chat_id[beat])
            chat_id_to_class = defaultdict(list)
            for classification, chat_ids in beat_to_class_to_chat_id[
                beat
            ].items():
                for id in chat_ids:
                    row = ss.data[ss.data[cell.CONV_ID] == id].iloc[0]
                    chat_id_to_class[id].append(classification)
                    chats_d["id"].append(id),
                    chats_d["class"].append(classification)
                    chats_d["user"].append(row["triggering_action_text"])
                    chats_d["agent"].append(row["message"])
            Results_cols[1].dataframe(pd.DataFrame(chats_d))
            # Results_cols[1].json(chat_id_to_class)

        expected_P = div_0_str(
            non_pooled_score[beat]["good_bot"], non_pooled_score[beat]["total"]
        )
        Results_cols[0].markdown(
            f"Measured: {beat}: {expected_P} = {non_pooled_score[beat]['good_bot']} / {non_pooled_score[beat]['total']}"
        )
        correct_calib = 0
        total_calib = 0
        for j in range(len(calib_accum_d_l)):
            correct = calib_accum_d_l[j]["Correct"][i]
            correct_calib += correct
            total = calib_accum_d_l[j]["Total"][i]
            total_calib += total
            calibration_P = div_0_str(correct, total)
            Results_cols[0].markdown(
                f"Mean calibration {j} {i} : {calibration_P} = {correct} / {total}"
            )
        try:
            AMA_pooled = (
                correct_calib
                / total_calib
                * non_pooled_score[beat]["good_bot"]
                / non_pooled_score[beat]["total"]
            )
        except ZeroDivisionError as e:
            AMA_pooled = 0.0
        calibration_P = div_0_str(correct_calib, total_calib)
        Results_cols[0].markdown(
            f"Expected AMA performance: {AMA_pooled:.1%} = {calibration_P} * {expected_P}"
        )


def show_partially_pooled(Context, accum_d, pooled_score, non_pooled_score):
    Context.markdown("## Partially Pooled")
    partial_pool_draws_df = cell.binomial_partial_pool(
        N=len(accum_d["Beat"]), K=accum_d["Total"], y=accum_d["Correct"]
    )

    # partial_pool_draws_df['phi_hat'] =\
    #     partial_pool_draws_df.apply(lambda row: row.phi * pooled_result, axis=1)

    # Partial Pooled graph
    Results_cols = Context.columns([3, 1])
    pooled_correct = sum(accum_d["Correct"])
    pooled_total = sum(accum_d["Total"])
    expected_P = div_0_str(pooled_correct, pooled_total)
    Results_cols[0].markdown(
        f"Pooled: {expected_P} = {pooled_correct} / {pooled_total}"
    )
    mean = statistics.mean(partial_pool_draws_df["phi"])
    plot = (
        p9.ggplot(
            partial_pool_draws_df, p9.aes(x="phi", y=p9.after_stat("count"))
        )
        + p9.geom_histogram()
        + p9.geom_vline(p9.aes(xintercept=mean))
        + p9.scale_x_continuous(limits=(0, 1))
        + p9.annotate(
            "text",
            x=mean,
            y=-20,
            label=f"{mean:.1%}",
            angle=0,
            size=40,
            color="#f409d5",
        )
    )
    Results_cols[1].pyplot(p9.ggplot.draw(plot))
    # Kappa
    Results_cols = Context.columns([3, 1])
    Results_cols[0].markdown("Ratio pooled/non-pooled")
    # mean = statistics.mean(partial_pool_draws_df['kappa'])
    plot = (
        p9.ggplot(
            partial_pool_draws_df, p9.aes(x="kappa", y=p9.after_stat("count"))
        )
        + p9.geom_histogram()
        + p9.geom_vline(p9.aes(xintercept=mean))
        # + p9.annotate("text", x=mean + .2, y=-20, label=f"{mean}", angle=0, size=40, color="#f409d5")
    )
    Results_cols[1].pyplot(p9.ggplot.draw(plot))

    for i in range(0, len(accum_d["Beat"])):
        expected_P = div_0_str(accum_d["Correct"][i], accum_d["Total"][i])
        mean = statistics.mean(partial_pool_draws_df[f"theta.{i + 1}"])
        plot = (
            p9.ggplot(
                partial_pool_draws_df,
                p9.aes(x=f"theta.{i + 1}", y=p9.after_stat("count")),
            )
            + p9.geom_histogram()
            + p9.geom_vline(p9.aes(xintercept=mean))
            + p9.scale_x_continuous(limits=(0, 1))
            + p9.annotate(
                "text",
                x=mean,
                y=-20,
                label=f"{mean:.1%}",
                angle=0,
                size=40,
                color="#f409d5",
            )
        )
        Results_cols = Context.columns([3, 1])
        Results_cols[1].pyplot(p9.ggplot.draw(plot))
        Results_cols[0].markdown(
            f'{accum_d["Beat"][i]}: {expected_P} = {accum_d["Correct"][i]} / {accum_d["Total"][i]}'
        )




def eval_on_existing(Context, page, i_cell):
    if "pooled_score" not in ss:
        ss.pooled_score = None
        ss.non_pooled_score = None
        ss.beat_to_class_to_chat_id = None
        ss.chat_id_to_beat_to_class = None
    if "beat_calibration_scores" not in ss:
        ss.beat_calibration_scores = []
        ss.accum_d = []
    if 'annot_coll_LLM' not in ss:
        ss.annot_coll_LLM = ss.annot_db[ss.collection_name]
    if 'eval_ids' not in ss:
        ss.eval_ids = list(ss.annot_coll_LLM.distinct('_id'))
    Control_cols = Context.columns(4)
    Control_cols[2].checkbox(
        "Show Pooled",
        value=True,
        key="pooled_cb",
        disabled=ss.pooled_score is None,
    )
    Control_cols[3].radio(
        "Show distributions or chats",
        options=["distributions", "chats"],
        index=1,
        key="dist_or_chat_r",
    )
    Control_cols[1].slider(
        "How many new chats?",
        min_value=0,
        max_value=100,
        value=1,
        key="number_of_chats_s",
        # disabled=not ss.apply_new_data_cb,
    )
    Control_cols[0].slider(
        "Repeat",
        min_value=1,
        max_value=100,
        value=1,
        key="number_of_repeat_evals_s",
    )
    Control_cols[0].slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=ss.temperature,
        on_change=update_ss_from_widget,
        args=["temperature", "temp_s"],
        key="temp_s",
    )
    Control_cols[1].slider(
        "Top-P/K",
        min_value=0.0,
        max_value=1.0,
        value=ss.top_p,
        on_change=update_ss_from_widget,
        args=["top_p", "top_p_s"],
        key="top_p_s",
    )
    Control_cols[1].slider(
        "Maximum number of tokens",
        min_value=1,
        max_value=100,
        value=ss.max_tokens,
        on_change=update_ss_from_widget,
        args=["max_tokens", "max_tokens_s"],
        key="max_tokens_s",
    )

    # 立生
    
    Control_cols[0].slider(
        "Evaluation size",
        value=min(10, len(ss.eval_ids)),
        min_value=1,
        max_value=len(ss.eval_ids),
        key="evaluation_size_sl",
    )

    if Control_cols[0].button(
        f"Evaluate classifier on {ss.evaluation_size_sl} annotations and apply to {ss.number_of_chats_s} new chats:"
    ):
        Header_cols = Context.columns(2)
        Header_cols[0].markdown("Counts")
        Header_cols[1].markdown("Certainty")
        beats_to_use = get_beats_to_use(i_cell)
        # collect results for new data
        # if ss.apply_new_data_cb:
        ss.accum_d = []
        ss.beat_calibration_scores = []
        (
            ss.pooled_score,
            ss.non_pooled_score,
            ss.beat_to_class_to_chat_id,
            ss.chat_id_to_beat_to_class,
        ) = process_new_data(
            page,
            i_cell,
            beats_to_use,
            num_chats=ss.number_of_chats_s,
            num_runs=ss.number_of_repeat_evals_s,
        )
        for j in range(ss.number_of_repeat_evals_s):
            beat_calibration_scores, accum_d = calibrate(
                page,
                i_cell,
                beats_to_use,
                temp=ss.temperature,
                top_p=ss.top_p,
                max_tokens=ss.max_tokens,
                count=ss.evaluation_size_sl,
            )
            ss.beat_calibration_scores.append(beat_calibration_scores)
            ss.accum_d.append(accum_d)

        # collect results for annotated data
        st.rerun()

    if ss.pooled_score is not None:
        print(f"accum_d: {ss.accum_d}")
        print(f"pooled_score: {ss.pooled_score}")
        show_pooled(Context, ss.accum_d, ss.pooled_score)

    # if ss.non_pooled_score is not None:
    #     show_non_pooled(
    #         Context,
    #         ss.accum_d,
    #         ss.non_pooled_score,
    #         ss.beat_to_class_to_chat_id,
    #         ss.chat_id_to_beat_to_class,
    #     )

    # if (
    #     ss.partially_pooled_cb
    #     and ss.non_pooled_score is not None
    #     and ss.pooled_score is not None
    # ):
    #     show_partially_pooled(
    #         Context, ss.accum_d[0], ss.pooled_score, ss.non_pooled_score
    #     )

    # Context.json(beat_scores)


def run_segment(
    seg_index,
    segments,
    page,
    i_cell,
    beats_to_use,
    temp,
    top_p,
    max_tokens,
):
    """Constructs and runs indicated segment against LLM
    arguments:
       seg_index: int what sub-segment of chat to run
       segments: list segments that can be run
       page: str page name in ss.config to pull setup from
       i_cell: int what cell withing page to access
       beats_to_use: str what beats to run
       temp: float probility temperature for LLM
       top_p: float probability top_p value for LLM
       max_tokens: maximum number of tokens that LLM can return
    """
    chat_data = segments[seg_index]

    # Calibrate.markdown(chat_data, unsafe_allow_html=True)
    prompt, ss.rubric = cell.assemble_prompt(
        ss.config[page][i_cell]["setup"], chat_data, beats_to_use
    )
    print(f"Prompt: {prompt}")

    print("#######Running")
    return cell.run_prompt(
        prompt, temperature=temp, top_p=top_p, max_tokens=max_tokens
    )


def run_eval(Calibrate, ss, page, i_cell):
    Select_cols = Calibrate.columns([3,1,2,2])
    options = [""] + ss.chats
    Select_cols[0].selectbox(
        "Pick a chat",
        options=options,
        index=options.index(ss.current_run_chat),
        on_change=select_chat,
        key=f"pick_a_chat_{ss.odd_even}_sb",
    )
    if ss[f"pick_a_chat_{ss.odd_even}_sb"] == "":
        return
    num_segments = len(ss.chat_segments)
    Select_cols[1].selectbox(
        f"Pick segment, {num_segments} exist",
        options=list(range(0, num_segments)),
        on_change=select_segment,
        key=f"segment_{ss.current_run_chat}_sb",
        disabled=num_segments == 1,
    )
    Select_cols[2].checkbox("Show prompt", value=False, key="show_prompt_cb")
    Select_cols[3].checkbox("Show segment", value=False, key="show_segment_cb")
    # Select_cols[2].checkbox("Show scores", value=True, key="show_scores_cb")

    beats_to_use = get_beats_to_use(i_cell)

    if ss.show_prompt_cb:
        prompt, rubric = cell.assemble_prompt(
            ss.config[page][i_cell]["setup"], "", beats_to_use
        )
        Calibrate.json(prompt)
    if ss.show_segment_cb:
        chat_data = ss.chat_segments[ss.current_seg_index]
        turns = [turn.strip() for turn in chat_data.split('\n') if turn != '']
        Calibrate.write(turns)
    if "run" not in ss:
        ss.run = False
    if "run_results" not in ss:
        ss.run_results = None

    Run_cols = Calibrate.columns([3, 1])
    Run_cols[1].slider(
        "Reruns", min_value=1, max_value=100, value=1, key="rerun_count"
    )

    if Run_cols[0].button(
        f"(Re)run current chat {ss.current_run_chat}, segment {ss.current_seg_index}, {ss.rerun_count} time(s)"
    ):
        ss.run = True

    if ss.run:  # forces run
        ss.run_results = []
        Calibrate.markdown(f"Running segment: {ss.current_seg_index}")
        for i in range(ss.rerun_count):
            print(f"{ss.indent}running {ss.current_run_chat} segment {ss.current_seg_index}")
            ss.response_json = run_segment(
                seg_index=ss.current_seg_index,
                segments=ss.chat_segments,
                page=page,
                i_cell=i_cell,
                beats_to_use=beats_to_use,
                temp=ss.temperature,
                top_p=ss.top_p,
                max_tokens=ss.max_tokens,
            )
            ss.run_results.append(ss.response_json)
        ss.last_run_seg_index = ss.current_seg_index
        ss.run = False
    if ss.run_results is not None:
        beat_to_classification_counts = defaultdict(lambda: defaultdict(int))
        for response_json in ss.run_results:
            response_text = response_json.choices[0].message.content
            print(f"Extracted LLM response: {response_text}")
            results = cell.extract_LLM_results(
                response_text, ss.config[page][i_cell]["setup"], beats_to_use
            )
            for beat, classification in results.items():
                beat_to_classification_counts[beat][classification] += 1
        beat_to_values = {}
        for setup_entry in ss.config[page][i_cell]["setup"]:
            if (
                setup_entry["include"]
                and setup_entry.get("type", "") == "reader"
            ):
                beat_to_values[setup_entry["beat"]] = setup_entry[
                    "possible values"
                ]
        #id = f"{ss.current_run_chat}_{ss.current_seg_index}"
        result = ss.annot_db[ss.collection_name].find_one(
                    {CID: int(ss.current_run_chat), SID: ss.current_seg_index})
        annotations = defaultdict(lambda: defaultdict(bool))
        if result is not None:
            annotations = json.loads(result[ANNOT])
        Header_cols = Calibrate.columns([1, 4, 2, 1])
        Header_cols[0].markdown("Beat")
        Header_cols[1].markdown("Prompt")
        Header_cols[2].markdown("LLM")
        Header_cols[3].markdown("Truth")
        # Header_cols[4].markdown("Totals")
        for beat, entry in ss.rubric.items():
            Ann_cols = Calibrate.columns([1, 4, 2, 1])
            if beat not in results:
                print(f"Skipping {beat}, not in results")
                Ann_cols[1].markdown(f"Skipping '{beat}', not in results")
                continue
            beat_annot = annotations.get(beat, "")
            opts = [""] + beat_to_values[beat]
            Ann_cols[0].markdown(beat)
            Ann_cols[1].markdown("\n ".join(entry))
            Ann_cols[2].json(beat_to_classification_counts[beat])
            Ann_cols[3].radio(
                " ",
                options=opts,
                index=opts.index(beat_annot),
                key=f"truth_{beat}_{ss.current_run_chat}_{ss.current_seg_index}",
            )
        Control_annots = Calibrate.columns(4)
        Control_annots[0].checkbox("Advance", value=True, key="advance_cb")
        Control_annots[2].text_input(
            "Notes:",
            value=annotations.get("Notes", ""),
            key=f"notes_{ss.current_run_chat}_{ss.current_seg_index}_ti",
        )
        if Control_annots[1].button("Save Annotation"):
            for beat, entry in ss.rubric.items():
                annotations[beat] = ss[
                    f"truth_{beat}_{ss.current_run_chat}_{ss.current_seg_index}"
                ]
            ss.annotator = "anonymous" if ss.username is None else ss.username
            annotations["Notes"] = ss[
                f"notes_{ss.current_run_chat}_{ss.current_seg_index}_ti"
            ]
            annotations["Annotator"] = ss.annotator
            annotations["Data"] = ss.chat_segments[ss.current_seg_index]
            annotations[
                "Session ID"
            ] = f"{ss.current_run_chat}_{ss.current_seg_index}"
            id = ss.annotator + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
            annotations["id"] = id
            annots = {'$set': {ANNOT: json.dumps(annotations)}}
            query_filter = {CID: int(ss.current_run_chat), 
                            SID: ss.current_seg_index}
            result = ss.annot_db[ss.collection_name].update_one(query_filter, annots, upsert=True)
            if result.upserted_id is not None:
                Calibrate.markdown("Updated annotation")
            else:
                Calibrate.markdown("Added annotation")
            if ss.advance_cb:
                ss.current_seg_index += 1
                if ss.current_seg_index == len(ss.chat_segments):
                    current_chat_index = ss.chats.index(ss.current_run_chat)
                    ss.current_run_chat = ss.chats[current_chat_index + 1]
                    print(
                        f"out of segments for chat, moving to ss.current_run_chat"
                    )
                    ss.current_seg_index = 0
                ss.run_results = None
                ss.run = True
                stage_segment(ss.current_run_chat, ss.current_seg_index)
            st.rerun()


def prompter(page, i_cell):
    ss = st.session_state
    ss.indent = "        "
    if "current_run_chat" not in ss:
        ss.current_run_chat = ''
    if "filtered_data" not in ss or ss.filtered_data is None:
        st.error("No ss.filtered_data defined, check config or add")
        st.stop()
    if "temperature" not in ss:
        ss.temperature = 1.0
    if "top_p" not in ss:
        ss.top_p = 0.0
    if "max_tokens" not in ss:
        ss.max_tokens = 1
    if "odd_even" not in ss:
        ss.odd_even = 0
    ss.odd_even = ss.odd_even + 1 % 2
    if "chats" not in ss:
        ss.chats = sorted(list(ss.filtered_data[CID].unique()))
    if "data_map" not in ss:
        ss.data_map = ss.config[page][i_cell]["data_map"]
    if "setup" not in ss:
        ss.setup = ss.config[page][i_cell]["setup"]
    Edit_run_cols = st.columns(3)

    if Edit_run_cols[0].checkbox("Edit", value=False, key="enable_edit_cb"):
        edit_prompts_ui(ss, page, i_cell)
    if Edit_run_cols[1].checkbox("Calibrate/Eval", value=False, 
                                key="enable_calibrate_cb"):
        Calibrate = st.container()
        eval_on_existing(Calibrate, page, i_cell)

    if Edit_run_cols[2].checkbox(
        "Run/Annotate", value=False, key="enable_run_ann_cb"
    ):
        run_eval(st, ss, page, i_cell)

        


# run_controls(i_cell, chats, Edit_exp)
# run 1 + annotate (Lishing's method)
# run all + annotate for next
# run against saved for group

# st.json(ss.p_d)


if __name__ == "__main__":

    if "nothing_to_save" not in ss:
        ss.nothing_to_save = True
    if "view_chat_df" not in ss:
        ss.view_chat_df = None
    st.info("Running main in LLM_prompter")
    print("****Running Stand Alone****")
    import pandas as pd
    import cell_util as cell

    import configure_from_url
    import load_data
    import cell_util as cell
    import col_val_filter

    SS = UI.session_state
    configure_from_url.get_config_file(UI)
    if "data" not in SS:
        load_data.load(UI, "data/AMA")
    if "filtered_data" not in SS:
        col_val_filter.run_col_val_filter_row(2)

    st.markdown(f"DB has {len(db_util.SPEAKPEEK_DB.keys())} keys")
    if st.button("load annotations"):
        # loaded_data_paths =
        loaded_data_paths = [
            #            "../docs/AMA/AMA Evaluation Sheet.xlsx",
            # "data/AMA_classified/AMA Annotation Data 0401-0531.xlsx",
            "data/AMA_classified/short.xlsx"
        ]

        # beats = cell.assemble_prompt(ss.config[page][i_cell]["setup"], "", None)[1]
        counter = 0
        for loaded_data_path in loaded_data_paths:
            annots_df = pd.read_excel(loaded_data_path)
            ss.annotator = f"EXCEL: {loaded_data_path}"
            if not (db_util.ANNOTATORS_DB.exists(ss.annotator)):
                db_util.ANNOTATORS_DB.set(ss.annotator, "")
            for i, annot_row in annots_df.iterrows():
                annotations = {}
                if (
                    not isinstance(annot_row["Session ID"], str)
                    or len(annot_row["Session ID"]) < 10
                ):  # catching too short or Nan
                    continue
                # for i, (num, excel_beat) in enumerate(
                #     beat_num_to_beat_label.items()
                # ):
                #     if annot_row[excel_beat] == 1.0:
                #         annot = True
                #     elif annot_row[excel_beat] == 0.0:
                #         annot = False
                #     else:
                #         continue
                #     annotations[excel_beat] = annot
                annotations["Notes"] = annot_row["Notes"]
                annotations["Annotator"] = ss.annotator
                data = [
                    {"role": "user", "content": annot_row["Utterance"]},
                    {"role": "assistant", "content": annot_row["Answer"]},
                ]
                annotations["Data"] = data
                annotations["Session ID"] = annot_row["Session ID"]
                # id = ss.annotator + "_" + datetime.now().strftime("%Y%m%d%H%M%S")
                annotations["id"] = annot_row["Session ID"]
                db_util.SPEAKPEEK_DB.set(
                    annot_row["Session ID"], json.dumps(annotations)
                )
                counter += 1
            print("Loaded {counter} from {loaded_data_path}")

    # st.dataframe(ss.filtered_data)
    # st.json(ss.config)
    if "chat_prompts" not in ss:
        cell.get_chat_prompts(ss, 3)
    prompter(st, 4)

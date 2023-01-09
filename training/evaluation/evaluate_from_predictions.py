#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team and OFA-X team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
import functools
import json
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig

from test_nlg import get_nlg_scores
from utils import checkpoint_utils
from utils.eval_utils import eval_step, merge_results, decode_fn

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

def clean(st):
    # remove everything in between parentheses
    return re.sub(r" \([^)]*\)", "", st)

def make_answer_dict(ref_string, task="vqa_gen_x"):
    if task == "vqa_gen_x":
        return {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in ref_string.split('&&')}
    elif task == "e_snli_ve":
        if ref_string == "neutral":
            return {"maybe": 1.0}
        elif ref_string == "entailment":
            return {"yes": 1.0}
        else:
            return {"no": 1.0}
    elif task == "vcr":
        return {ref_string: 1.0}


def main(cfg: DictConfig, task=None, exclude_bounding_boxes=False, **kwargs):
    # load predictions from json file with lists for ids, answers, predictions, and ground truth explanations
    output_path = os.path.join(cfg.common_eval.results_path, "{}_predict.json".format(cfg.dataset.gen_subset))
    with open(output_path, 'r') as fw:
        results = json.load(fw)
    df = pd.read_csv(cfg.common_eval.path, sep='\t')
    if task == "vqa_gen_x":
        df.columns = ["id", "image_id", "question", "answer", "explanation", "o", "image"]
    elif task in ["e_snli_ve", "vcr"]:
        df.columns = ["id", "image_id", "image", "question", "explanation", "answer"]

    ids = []
    target_expls = []
    expls = []
    correct_ids = []
    task_score = 0
    correct_expls = []
    correct_target_expls = []

    # Groupby because each id has multiple ground truth explanations
    for id, indices in df.groupby("id").groups.items():
        ids += [id]
        # Get the ground truth answers and explanations
        tgt_expls = [df["explanation"].iloc[i] for i in indices]
        if exclude_bounding_boxes:
            tgt_expls = [clean(expl) for expl in tgt_expls]
        target_expls += [tgt_expls]
        target_answers = make_answer_dict(df["answer"].iloc[indices[0]], task=task)

        # Get the predicted explanations
        index = results["id"].index(str(id))
        expl = results["explanation"][index].lstrip("because ")
        expls += [expl]
        ans = results["answer"][index]

        # Check if the predicted answer is correct
        if ans in target_answers:
            if exclude_bounding_boxes:
                expl = clean(expl)
            correct_expls += [expl]
            correct_target_expls.append(tgt_expls)
            correct_ids += [id]
            task_score += target_answers[ans]

    all_scores = get_nlg_scores(expls, target_expls, device=cfg.distributed_training.device_id)
    correct_scores = get_nlg_scores(correct_expls, correct_target_expls, device=cfg.distributed_training.device_id)

    # Print scores
    accuracy = len(correct_ids) / len(ids)
    task_score_final = task_score / len(ids)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Task Score: {task_score_final:.4f}")
    print("Scores for all explanations ('unfiltered'):")
    for key, value in all_scores.items():
        print(f"{key}: {value:.4f}")
    print("Scores for correct explanations ('filtered'):")
    for key, value in correct_scores.items():
        print(f"{key}: {value:.4f}")
    print("Scores scaled by accuracy:")
    for key, value in list(all_scores.items()):
        print(f"{key}: {value * len(correct_ids) / len(ids):.4f}")
        # Scale scores by accuracy (equivalent to scoring 0 for every incorrect example)
        all_scores[f"scaled_{key}"] = value * len(correct_ids) / len(ids)
        # Add filtered scores to dict for saving
        all_scores[f"filtered_{key}"] = correct_scores[key]
        # Rename unfiltered scores
        all_scores[f"unfiltered_{key}"] = value
    all_scores["accuracy"] = accuracy
    all_scores["task_score"] = task_score_final
    output_path = os.path.join(cfg.common_eval.results_path, "{}_scores.json".format(cfg.dataset.gen_subset))
    with open(output_path, 'w') as fw:
        json.dump(all_scores, fw)


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--exclude-bounding-boxes", type=bool, default=False)
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    if cfg.task._name == "vqa_gen_x":
        distributed_utils.call_main(
            cfg, main, task=cfg.task._name, beam_search_vqa_eval=True, zero_shot=True
        )
    else:
        exclude_bounding_boxes = args.exclude_bounding_boxes
        distributed_utils.call_main(cfg, main, task=cfg.task._name, exclude_bounding_boxes=exclude_bounding_boxes)


if __name__ == "__main__":
    cli_main()

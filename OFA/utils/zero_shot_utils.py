# Copyright 2022 The OFA-Sys Team and OFA-X team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import string
import math

import torch

from data import data_utils


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


def eval_vqa_gen(task, generator, models, sample, use_prefix_tokens=True, **kwargs):
    if use_prefix_tokens:
        prefix_length = len(sample['prefix_tokens'][0])
        hypos = task.inference_step(generator, models, sample, prefix_tokens=sample['prefix_tokens'])
    else:
        prefix_length = 0
        hypos = task.inference_step(generator, models, sample)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(hypos[i][0]["tokens"][prefix_length:], task.tgt_dict, task.bpe, generator)
        results.append({
            "question_id": sample_id,
            "answer": detok_hypo_str.strip(),
            "attention": hypos[i][0]["attention"],
            "tokens": hypos[i][0]["tokens"]
        })
    #scores = [ref_dict.get(result['answer'], 0) for ref_dict, result in zip(sample['ref_dict'], results)]
    return results #, scores


def zero_shot_step(task, generator, models, sample, **kwargs):
    generator.zero_shot = True
    if task.cfg._name in ['vqa_gen', 'vqa_gen_x']:
        generator.constraint_trie = None
        return eval_vqa_gen(task, generator, models, sample, **kwargs)
    else:
        raise NotImplementedError

# Copyright 2022 The OFA-Sys Team and OFA-X team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.
import functools
import json
import logging
import math
import random
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional

import torch
import wandb
from datasets import load_metric

from fairseq import metrics
from fairseq.tasks import register_task
import torchvision.transforms.functional as TF
import torchmetrics.functional as eval_metrics
from torchmetrics.text.bert import BERTScore

from tasks.ofa_task import OFAConfig, OFATask
from data.mm_data.vcr_baseline_dataset import VCRBaselineDataset
from data.file_dataset import FileDataset
from data import data_utils
from utils import eval_utils
from utils.trie import Trie

logger = logging.getLogger(__name__)

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


def aggregate_metrics(examples):
    # Aggregate metrics from all samples
    agg_metrics = {}
    for k in examples[0].keys():
        agg_metrics[f"_{k}_sum"] = sum([m[k] for m in examples])
        agg_metrics[f"_{k}_cnt"] = len(examples)
    return agg_metrics


@dataclass
class VCRBaseConfig(OFAConfig):
    ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1, "maybe": 2}',
        metadata={"help": 'answer to label dict'},
    )
    add_caption: bool = field(
        default=False,
        metadata={"help": "add caption to encoder"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "prompt_type"},
    )
    eval_args: Optional[str] = field(
        default='{"beam":1,"unnormalized":true,"temperature":1.0}',
        metadata={
            "help": 'generation args as JSON string for inference, only activated when --val-inference-type=beamsearch'
        },
    )


@register_task("vcr_baseline", dataclass=VCRBaseConfig)
class VCRBaseTask(OFATask):
    def __init__(self, cfg: VCRBaseConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = VCRBaselineDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            add_caption=self.cfg.add_caption,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            prompt_type=self.cfg.prompt_type
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        gen_args = json.loads(self.cfg.eval_args)
        gen_args["match_source_len"] = False
        self.generator = self.build_generator(
            [model], Namespace(**gen_args)
        )
        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
        return seq_generator

    def valid_step(self, sample, model, criterion, table=None, **extra_kwargs):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        model.eval()
        with torch.no_grad():
            hyps, raw_hyps = eval_utils.eval_step(
                task=self,
                generator=self.generator,
                models=[model],
                sample=sample
            )

        scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]

        logging_output["_snli_score_sum"] = sum(scores)
        logging_output["_snli_cnt"] = len(scores)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters, sum_key, cnt_key):
            score = meters[sum_key].sum / meters[cnt_key].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        def derived_metrics(count_key, sum_key, log_key):
            if sum_logs(count_key) > 0:
                metrics.log_scalar(sum_key, sum_logs(sum_key))
                metrics.log_scalar(count_key, sum_logs(count_key))
                metrics.log_derived(log_key, functools.partial(compute_score, sum_key=sum_key, cnt_key=count_key))

        counts = [k for k in logging_outputs[0].keys() if k.endswith("_cnt")]
        sums = [k for k in logging_outputs[0].keys() if k.endswith("_sum")]
        log_names = [f"{k.split('_')[1]}_score" for k in counts]
        for count_key, sum_key, log_key in zip(counts, sums, log_names):
            derived_metrics(count_key, sum_key, log_key)

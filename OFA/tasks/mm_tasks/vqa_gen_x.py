# Copyright 2022 The OFA-Sys Team and OFA-X team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.
import functools
import random
from dataclasses import dataclass, field
import json
import logging
from functools import partial
from typing import Optional
from argparse import Namespace

import torchvision.transforms.functional as TF
import wandb
from datasets import load_metric

from data.file_dataset import FileDataset

import torch
from fairseq import metrics
from fairseq.tasks import register_task
import torchmetrics.functional as eval_metrics

from data.mm_data.vqa_gen_x_dataset import VqaGenXDataset
from tasks.ofa_task import OFAConfig, OFATask
from utils import eval_utils

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

def calculate_expl_metrics(expl, target, bert_metric=None):
    # Calculate metrics: BLEU, ROUGE, BERTScore for one sample
    final_metrics = {}
    bleu = eval_metrics.bleu_score(expl, [target])
    rouge = eval_metrics.text.rouge_score(expl, target, rouge_keys=("rougeL"))
    if bert_metric is not None and expl != "":
        bert_metric.add_batch(predictions=[expl], references=[[target]])
        bert_score = bert_metric.compute(
            model_type="distilbert-base-uncased"
        )
        final_metrics["BERTScore"] = torch.tensor(bert_score["f1"]).mean()
    else:
        final_metrics["BERTScore"] = torch.tensor(0.0)
    final_metrics.update({
        "bleu": bleu,
        "rouge-fmeasure": rouge['rougeL_fmeasure'],
        "rouge-precision": rouge['rougeL_precision'],
        "rouge-recall": rouge['rougeL_recall'],
    })
    return final_metrics

@dataclass
class VqaGenXConfig(OFAConfig):
    max_object_length: int = field(
        default=30, metadata={"help": "the maximum object sequence length"}
    )
    add_object: bool = field(
        default=False,
        metadata={"help": "add object to encoder"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "prompt_type"},
    )
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    val_inference_type: Optional[str] = field(
        default='allcand',
        metadata={"help": "inference type in validation (allcand or beamsearch), default to allcand"},
    )    
    eval_args: Optional[str] = field(
        default='{"beam":5,"unnormalized":true,"temperature":1.0}',
        metadata={
            "help": 'generation args as JSON string for inference, only activated when --val-inference-type=beamsearch'
        },
    )    


@register_task("vqa_gen_x", dataclass=VqaGenXConfig)
class VqaGenXTask(OFATask):
    def __init__(self, cfg: VqaGenXConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.bert_metric = load_metric(
            "bertscore",
            experiment_id=str(random.randrange(999999)),
        )

        self.uses_ema = self.cfg.uses_ema

        assert self.cfg.val_inference_type in ["allcand", "beamsearch", "beamsearch+similarity"], \
            "Unknown inference type encountered: {}, should be allcand or beamsearch.".format(self.cfg.val_inference_type)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            table_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            table_path = paths[-1]
        dataset = FileDataset(table_path, self.cfg.selected_cols)

        self.datasets[split] = VqaGenXDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_object_length=self.cfg.max_object_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            add_object=self.cfg.add_object,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            prompt_type=self.cfg.prompt_type
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.val_inference_type == "beamsearch":
            gen_args = json.loads(self.cfg.eval_args)
            self.generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        elif self.cfg.val_inference_type == "beamsearch+similarity":
            gen_args = json.loads(self.cfg.eval_args)
            gen_args["match_source_len"] = False
            self.generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        else:
            raise NotImplementedError("Error: Unknown inference type encountered.")

        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)

        return seq_generator

    def valid_step(self, sample, model, criterion, table=None, **extra_kwargs):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        decode = functools.partial(decode_fn, tgt_dict=self.tgt_dict, bpe=self.bpe, generator=self.generator)
        if self.uses_ema:
            assert 'ema_model' in extra_kwargs and extra_kwargs['ema_model'] is not None
        if self.uses_ema:
            eval_model = extra_kwargs['ema_model']
        else:
            eval_model = model

        eval_model.eval()
        with torch.no_grad():
            if self.cfg.val_inference_type in ["beamsearch", "beamsearch+similarity"]:
                hyps, expls, raw_hyps = eval_utils.eval_step(
                    task=self,
                    generator=self.generator,
                    models=[eval_model],
                    sample=sample
                )
            else:
                raise NotImplementedError("Error: Unknown inference type encountered.")

        questions = [decode(x[x.ne(1)]).strip() for x in sample["net_input"]["src_tokens"]]
        scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
        target_expls = [decode(x[x.ne(1)]).strip() for x in sample["explanations"]]
        val_metrics_split = [
            calculate_expl_metrics(expl, target_expl, self.bert_metric)
            for expl, target_expl in zip(expls, target_expls)
        ]
        val_metrics = aggregate_metrics(val_metrics_split)
        expl_scores = [m["BERTScore"] for m in val_metrics_split]

        # Log samples to wandb
        table_data = zip(questions, raw_hyps, hyps, expls, sample['ref_dict'], target_expls, sample['net_input']['patch_images'])
        for i, (q, raw, ans, expl, ref_dict, target_expl, image) in enumerate(table_data):
            hypothesis = raw[0]["tokens"]
            # remove padding from decoder prompt
            prefix_len = sample['prefix_tokens'][i].ne(1).sum().item()
            hypothesis = hypothesis[prefix_len:]
            hypothesis_str = decode(hypothesis).strip()
            target_ans = f"{ref_dict}"
            image = TF.resize(image, [128, 128])
            im = wandb.Image(image)
            table.add_data(im, q, hypothesis_str, ans, expl, target_ans, target_expl)

        logging_output["_vqa_score_sum"] = sum(scores)
        logging_output["_vqa_cnt"] = len(scores)
        logging_output["_expl_score_sum"] = sum(expl_scores)
        logging_output["_expl_cnt"] = len(expl_scores)
        logging_output["_total_score_sum"] = sum([scores[i] * expl_scores[i] for i in range(len(scores))])
        logging_output["_total_cnt"] = len(scores)

        logging_output = dict(logging_output, **val_metrics)

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
                metrics.log_derived(log_key, partial(compute_score, sum_key=sum_key, cnt_key=count_key))

        counts = [k for k in logging_outputs[0].keys() if k.endswith("_cnt")]
        sums = [k for k in logging_outputs[0].keys() if k.endswith("_sum")]
        log_names = [f"{k.split('_')[1]}_score" for k in counts]
        for count_key, sum_key, log_key in zip(counts, sums, log_names):
            derived_metrics(count_key, sum_key, log_key)
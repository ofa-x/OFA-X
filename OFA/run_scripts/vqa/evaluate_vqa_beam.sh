#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8082
src_dir="../../"
export PYTHONPATH="${src_dir}:${src_dir}/fairseq:${PYTHONPATH}"

user_dir=../../ofa_module
bpe_dir=../../utils/BPE


data=../../../data/vqax/test_x.tsv
ans2label_file=../../../data/vqax/trainval_ans2label.pkl
path=../../checkpoints/vqa_large_best.pt
result_path=../../../results/vqa_test_beam
selected_cols=0,6,2,3,4

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=test \
    --results-path=${result_path} \
    --fp16 \
    --ema-eval \
    --unconstrained-training \
    --beam-search-vqa-eval \
    --beam=1 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}"
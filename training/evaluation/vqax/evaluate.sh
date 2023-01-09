#!/usr/bin/env bash

# Use as 'bash evaluate.sh <checkpoint_path> <run_name>'

src_dir="../OFA"
export PYTHONPATH="${src_dir}:${src_dir}/fairseq:${PYTHONPATH}"
export MASTER_PORT=8182

user_dir=../OFA/ofa_module
bpe_dir=../OFA/utils/BPE

path="$1"
run_name="$2"
data="$3"
result_path=../results/${run_name}
selected_cols=0,6,2,3,4,5
valid_batch_size=64

python -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ./evaluation/evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen_x \
    --batch-size=64 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset="test" \
    --results-path=${result_path} \
    --ema-eval \
    --beam-search-vqa-eval \
    --beam=1 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"valid_batch_size\":\"${valid_batch_size}\"}"
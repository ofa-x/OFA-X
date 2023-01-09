#!/usr/bin/env bash

# Use as 'bash evaluate.sh <checkpoint_path> <run_name>'

src_dir="../OFA"
export PYTHONPATH="${src_dir}:${src_dir}/fairseq:${PYTHONPATH}"
export MASTER_PORT=8183

user_dir=../OFA/ofa_module
bpe_dir=../OFA/utils/BPE


run_name="$1"
data="$2"
exclude_bounding_boxes="$3"
result_path=../results/${run_name}
selected_cols=0,2,3,4,5
valid_batch_size=16

python -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ./evaluation/evaluate_from_predictions.py \
    ${data} \
    --path=${data} \
    --user-dir=${user_dir} \
    --task=vcr \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset="test" \
    --exclude-bounding-boxes=${exclude_bounding_boxes} \
    --results-path=${result_path} \
    --beam=1 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"valid_batch_size\":\"${valid_batch_size}\"}"
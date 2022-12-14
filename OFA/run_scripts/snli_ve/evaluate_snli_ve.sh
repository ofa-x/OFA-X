#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7081

src_dir="../../"
export PYTHONPATH="${src_dir}:${src_dir}/fairseq:${PYTHONPATH}"

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=test

data=../../../data/esnlive/esnlive_test.tsv
path=../../checkpoints/snli_ve_large_best.pt
result_path=../../../results/esnlive_test
selected_cols=0,2,3,4,5

python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=snli_ve \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
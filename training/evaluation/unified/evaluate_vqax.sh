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
valid_batch_size=32

python -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ./evaluation/evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen_x \
    --batch-size=32 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset="test" \
    --results-path=${result_path} \
    --zero-shot \
    --fp16 \
    --bpe-dir=${bpe_dir} \
    --beam-search-vqa-eval  \
    --val-inference-type="beamsearch+similarity"  \
    --prompt-type="without_decoder_prompt" \
    --beam=1 \
    --selected-cols=${selected_cols} \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0 \
    --eval-args="{\"beam\":1,\"unnormalized\":\"True\",\"temperature\":1.0}" \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"valid_batch_size\":\"${valid_batch_size}\"}"


python -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ./evaluation/evaluate.py
${data}
--path ${path}
--user-dir ${user_dir}
--task vqa_gen_x
--batch-size 32
--log-format simple --log-interval 10
--seed 7
--gen-subset "test"
--results-path ${result_path}
--zero-shot
--fp16
--bpe-dir ${bpe_dir}
--beam-search-vqa-eval
--val-inference-type "beamsearch+similarity"
--prompt-type "without_decoder_prompt"
--beam 1
--selected-cols ${selected_cols}
--unnormalized
--temperature 1.0
--num-workers 0
--eval-args "{\"beam\":1,\"unnormalized\":\"True\",\"temperature\":1.0}"
--model-overrides "{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"valid_batch_size\":\"${valid_batch_size}\"}
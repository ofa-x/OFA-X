#!/usr/bin/env bash
# Run this as 'bash run_training.sh <config_file> <run_name> <checkpoint_path>' from the training folder
src_dir="../OFA"

export PYTHONPATH="${src_dir}:${src_dir}/fairseq:${PYTHONPATH}"

run_name_placeholder="vcr_training_run"
restore_checkpoint_file_placeholder="restore_checkpoint_file"

config_file="$1"
run_name="$2"
checkpoint_path="$3"

echo "Starting training with config file: ${config_file} and run name: ${run_name}"

config="$(cat "${config_file}")"
config="${config//$run_name_placeholder/$run_name}"
config="${config//$restore_checkpoint_file_placeholder/$checkpoint_path}"

mkdir -p ../outputs/vcr_logs
mkdir ../outputs/vcr_checkpoints
# shellcheck disable=SC2046
python -m torch.distributed.launch ${config}
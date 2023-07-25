#!/bin/bash
set -x
set -u


now=$(date +"%Y%m%d_%H%M%S")
jobname="train-$1-$now"
echo "job name is $jobname"

config_file=$2
mkdir -p log
mkdir -p logs/${jobname}
cp ${config_file} logs/${jobname}

export CUDA_VISIBLE_DEVICES=0
python custom_train.py --cfg_path ${config_file} \
  --num_gpus 1 --num_nodes 1 \
  --ckpt_path ckpts/train-light_test-20230723_144459/{epoch:d}/epoch=0-step=189999.ckpt \
  --num_epochs 20 --batch_size 2048 --test_batch_size 512 --num_workers 16 \
  --exp_name ${jobname} 2>&1|tee log/${jobname}.log \
  
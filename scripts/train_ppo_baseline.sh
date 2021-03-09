#!/bin/sh
export BABYAI_STORAGE='storage'
eval "$(conda shell.bash hook)"
conda activate ella
python babyai/scripts/train_rl.py \
--arch expert_filmcnn \
--env $1 \
--hrl vanilla \
--log-interval 1 --save-interval 15  --val-interval 15 --val-episodes 128 \
--procs 64 --frames-per-proc 40 --recurrence 20 \
--seed $3 \
--model $2-PPO-NoPre-$3 \
#--wb

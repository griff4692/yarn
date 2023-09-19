#!/bin/bash

GUIDANCE=$1
OUTPUT_DIR="/nlp/projects/summarization/bhc_data_cleanup/bhc_weights/yarn-7b-8k-$1"
echo $OUTPUT_DIR
YARN_FACTOR=$2
CKPT=$3

accelerate launch finetune.py \
    --wandb yarn \
    --deepspeed \
    --output-dir $OUTPUT_DIR \
    --model NousResearch/Llama-2-7b-hf \
    --yarn-factor $YARN_FACTOR \
    --max-train-steps 5000 \
    --warmup-steps 100 \
    --gradient-accumulate-every 32 \
    --checkpointing-steps 250 \
    --guidance $GUIDANCE \
    --resume-from-checkpoint $CKPT \

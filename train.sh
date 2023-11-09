#!/bin/bash

EXPERIMENT=$1
GUIDANCE=$2
OUTPUT_DIR="/nlp/projects/summarization/bhc_data_cleanup/bhc_weights/yarn/$EXPERIMENT"
echo $OUTPUT_DIR
#CKPT=$3

#accelerate launch finetune.py \
#    --wandb yarn \
#    --deepspeed \
#    --output-dir $OUTPUT_DIR \
#    --model NousResearch/Llama-2-7b-hf \
#    --yarn-factor $YARN_FACTOR \
#    --max-train-steps 5000 \
#    --warmup-steps 100 \
#    --gradient-accumulate-every 32 \
#    --checkpointing-steps 250 \
#    --guidance $GUIDANCE \
#    --resume-from-checkpoint $CKPT \


accelerate launch finetune.py \
    --output-dir $OUTPUT_DIR \
    --model mistralai/Mistral-7B-v0.1 \
    --architecture mistral \
    --scaling-factor 8 \
    --max-position-embeddings 16384 \
    --lr-schedule constant \
    --guidance $GUIDANCE \
    --learning-rate 0.000001 \
    --max-train-steps 1000 \
    --gradient-accumulate-every 32 \
    --checkpointing-steps 100

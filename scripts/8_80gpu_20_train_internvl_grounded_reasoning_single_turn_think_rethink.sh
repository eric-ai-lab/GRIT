#!/bin/bash

setting='h100_10_vsr_internvl_add_grounded_reasoning_single_turn_think_rethink'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=$setting

# Load config variables
source scripts/train_base_config.sh

# Run the training script with DeepSpeed
python -m accelerate.commands.launch \
    --config_file ./accelerate_configs/deepspeed_zero2.yaml \
    --main_process_port 20092 \
    grpo-gr/GRPO_GR.py \
    --train_data_path ./GRIT_data/tallyqa_train_10.jsonl,./GRIT_data/vsr_cot_train_10.jsonl \
    --train_image_folder_path ./GRIT_data/tallyqa,./GRIT_data/vsr \
    --eval_data_path ./GRIT_data/vsr_val.jsonl,./GRIT_data/mme_val.jsonl,./GRIT_data/tallyqa_val.jsonl,./GRIT_data/gqa_val.jsonl,./GRIT_data/mathvista_mini_val.jsonl,./GRIT_data/ovd_position_val.jsonl \
    --eval_image_folder_path ./GRIT_data/vsr,./GRIT_data/mme,./GRIT_data/tallyqa,./GRIT_data/gqa,./GRIT_data/mathvista_mini,./GRIT_data/ovd_position \
    --setting $setting \
    --max_turns 1 \
    --output_dir output/$setting \
    --hub_model_id $setting \
    $COMMON_ARGS \
    --eval_steps 50 \
    --save_steps 50 \
    --num_train_epochs 200 \
    --lr_scheduler_type cosine \
    --per_device_eval_batch_size 96 \
    --model_name_or_path OpenGVLab/InternVL3-2B-Instruct\
    --max_prompt_length 1500
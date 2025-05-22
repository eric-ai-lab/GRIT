# config.sh

COMMON_ARGS="
    --project_root_path $(pwd) \
    --python_path_for_dino $(which python) \
    --dataset_name rr \
    --learning_rate 2e-6 \
    --num_generations 4 \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --torch_empty_cache_steps 1 \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --max_completion_length 1000 \
    --max_prompt_length 750 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --num_train_epochs 10\
    --save_total_limit 10 \
    --report_to wandb \
    --logging_steps 1 \
    --beta 0.01 \
    --bf16 True \
    --bf16_full_eval True \
    --log_completions True\
    --push_to_hub True 
"

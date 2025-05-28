#!/bin/bash

# # 清除系统 CUDA 环境变量
# unset CUDA_HOME
# unset CUDA_PATH
# unset CUDA_ROOT
# unset LD_LIBRARY_PATH

# # 设置 PyTorch 使用自带的 CUDA
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
# export CUDA_HOME=$CONDA_PREFIX
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 8 --lora_alpha 16 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/mydata/llava_train_mixed.json \
    --image_folder ./playground/data/coco/train2017 \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-lora-debug \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

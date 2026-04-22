#!/bin/bash

# --- 1. 环境设置 ---
export MEGATRON_LM_PATH='/mnt/data/share-ssd/user/shaomingchen/code/long-audio/time-aware/ft_omni/Megatron-LM/' 
export HF_ENDPOINT=https://pkgs.d.xiaomi.net/artifactory/api/huggingfaceml/huggingface-remote
export MS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# PYTHONPATH 和 cuDNN
export PYTHONPATH=$MEGATRON_LM_PATH:$PYTHONPATH 
export LD_LIBRARY_PATH=/mnt/data/share-ssd/user/shaomingchen/miniconda3/envs/swift/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 【重要修正】环境变量名必须加 _CUDA_ 否则不生效
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# --- 2. 启动训练 ---
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --model /mnt/data/share-ssd/user/shaomingchen/code/long-audio/time-aware/ft_omni/Qwen3-Omni-30B-A3B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    \
    --dataset '/mnt/data/share-ssd/user/shaomingchen/code/long-audio/time-aware/ft_omni/ms-swift/examples/train/multimodal/data/train/train.jsonl' \
    --val_dataset '/mnt/data/share-ssd/user/shaomingchen/code/long-audio/time-aware/ft_omni/ms-swift/examples/train/multimodal/data/train/val.jsonl' \
    \
    --tensor_model_parallel_size 1 \
    --sequence_parallel false \
    \
    --freeze_vit true \
    --freeze_aligner true \
    \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    \
    --tensor_model_parallel_size 1 \
    --sequence_parallel false \
    --expert_model_parallel_size 4 \
    \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    \
    --finetune true \
    --cross_entropy_loss_fusion true \
    \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    \
    --max_length 32768 \
    \
    --max_epochs 3 \
    --save_interval 800 \
    --eval_interval 800 \
    \
    --save /mnt/data/share-oss/user/shaomingchen/ckpt/long-audio/time-aware/ceshi/1-21-test-megatron \
    \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot \
    \
    --bf16 true \
    2>&1 | tee "$OUTPUT_DIR/training_8gpu.log"
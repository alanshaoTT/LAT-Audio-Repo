#!/bin/bash

# ==========================================
# CloudML 专用启动脚本 (2机16卡 + 日志管理)
# ==========================================

# --- 1. 日志路径配置 (你的自定义需求) ---
# 定义日志存放目录
LOG_DIR="/mnt/data/share-ssd/user/shaomingchen/code/long-audio/time-aware/ft_omni/ms-swift/examples/train/multimodal/logs"

# 自动创建目录 (防止目录不存在报错)
mkdir -p "$LOG_DIR"

# 获取当前时间 (格式: 20260121_180800)
CURRENT_TIME=$(date "+%Y%m%d_%H%M%S")

# 定义最终日志文件名 (加入 RANK 以区分不同机器)
LOG_FILE="${LOG_DIR}/${CURRENT_TIME}_node_${RANK}.log"

echo "🚀 [CloudML] 任务启动..."
echo "节点 Rank: $RANK"
echo "日志将保存到: $LOG_FILE"

# --- 2. 环境设置 ---
export MEGATRON_LM_PATH='/mnt/data/share-ssd/user/shaomingchen/code/long-audio/time-aware/ft_omni/Megatron-LM/' 
export MS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="/tmp/shaomingchen/hf_cache"
export MODELSCOPE_CACHE="/tmp/shaomingchen/modelscope_cache"
export CUDA_VISIBLE_DEVICES=1,3
# 确保目录存在
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$MODELSCOPE_CACHE"

export PYTHONPATH=$MEGATRON_LM_PATH:$PYTHONPATH 
export LD_LIBRARY_PATH=/mnt/data/share-ssd/user/shaomingchen/miniconda3/envs/swift/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# 显存优化
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTORCH_ALLOC_CONF='expandable_segments:True'

# 定义 Checkpoint 保存路径 (OSS/JuiceFS)
OUTPUT_DIR="/mnt/data/share-oss/user/shaomingchen/ckpt/long-audio/time-aware/test"
mkdir -p "$OUTPUT_DIR"

# 获取 megatron 绝对路径
MEGATRON_BIN=$(which megatron)

# --- 3. 启动命令 (torchrun + tee 日志) ---
# --nproc_per_node=2 \
#    --nnodes=$WORLD_SIZE \
#    --node_rank=$RANK \
#    --master_addr=$MASTER_ADDR \
#    --master_port=$MASTER_PORT \
torchrun \
   --standalone \
   --nproc_per_node=2 \
    "$MEGATRON_BIN" sft \
    --model /mnt/data/share-ssd/user/liuchang32/.cache/modelscope/hub/models/Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset '/mnt/data/share-ssd/user/shaomingchen/code/long-audio/time-aware/ft_omni/ms-swift/examples/train/multimodal/data/train/train.jsonl' \
    --val_dataset '/mnt/data/share-ssd/user/shaomingchen/code/long-audio/time-aware/ft_omni/ms-swift/examples/train/multimodal/data/train/val.jsonl' \
    \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --expert_model_parallel_size 1 \
    \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    \
    --freeze_vit false \
    --freeze_aligner false \
    \
    --micro_batch_size 1 \
    --global_batch_size 24 \
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
    --save_interval 1000 \
    --eval_interval 1000 \
    \
    --save "$OUTPUT_DIR" \
    \
    --system 'You are a helpful assistant.' \
    --num_workers 8 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 64 \
    --model_author swift \
    --model_name swift-robot \
    --lazy_tokenize true \
    \
    --bf16 true \
    --no-gradient-accumulation-fusion \
    2>&1 | tee "$LOG_FILE"

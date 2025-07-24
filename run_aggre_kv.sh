#!/bin/bash

# 定义日志文件前缀和使用的 GPU
#TASK="GPQA"
GPU_ID=1
LOG_DIR="log/log_aggre_kv"
mkdir -p ${LOG_DIR}
# 定义种子值数组（根据需求扩展）
TASKS=("GPQA" "MATH_Hard")
SEEDS=(0)
BATCHES=(1 2 4 8 16)

mkdir -p ${LOG_DIR}
# 循环执行每个种子的任务
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
      for BATCH in "${BATCHES[@]}"; do

      echo "========================================"
      echo "开始实验：TASK = ${TASK} BATCH = ${BATCH}"
      echo "时间: $(date)"
      echo "========================================"

      AGGR_LOG="${LOG_DIR}/${TASK}_batch${BATCH}_seed${SEED}_kv.log"
      CUDA_VISIBLE_DEVICES=$GPU_ID python test_aggregator_kv.py \
      --aggregator LlamaR1 \
      --task $TASK  \
      --suffix $TASK \
      --batch $BATCH \
      --seed $SEED 2>&1 | tee -a "${AGGR_LOG}"
      done
    done
done

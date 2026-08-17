#!/usr/bin/env bash
set -e

for reward in llm_balanced llm_gated llm_ratio; do
  python main_train.py --drl td3 --reward "${reward}" --ep-num 2 --seed 0 \
    --trained-uav --uav-benchmark-reward see \
    --project-name "smoke/${reward}"
done


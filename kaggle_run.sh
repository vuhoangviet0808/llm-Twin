#!/usr/bin/env bash
set -e

EPISODES=300
EVAL_EPISODES=5
SEEDS=(0 1 2 3 4)
REWARDS=("see" "llm_balanced" "llm_gated" "llm_ratio")

for reward in "${REWARDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    project="llm_reward/${reward}_seed_${seed}"
    model_path="data/storage/${project}"
    echo "[Kaggle] ${reward}, seed ${seed}"
    if [[ -f "${model_path}/Actor_G_and_Phi_TD3" && -f "${model_path}/Critic_1_G_and_Phi_TD3" \
          && -f "${model_path}/Critic_2_G_and_Phi_TD3" && -f "${model_path}/Actor_UAV_TD3" \
          && -f "${model_path}/Critic_1_UAV_TD3" && -f "${model_path}/Critic_2_UAV_TD3" ]]; then
      echo "[Kaggle] training checkpoint exists; skipping training"
    else
      python main_train.py --drl td3 --reward "${reward}" --ep-num "${EPISODES}" \
        --seed "${seed}" --trained-uav --uav-benchmark-reward see \
        --project-name "${project}"
    fi
    if [[ -f "${model_path}/evaluation/evaluation_summary.json" ]]; then
      echo "[Kaggle] evaluation exists; skipping evaluation"
    else
      python evaluate_reward.py --model-path "${model_path}" --drl td3 \
        --reward "${reward}" --episodes "${EVAL_EPISODES}" --seed 100
    fi
  done
done

python summarize_experiments.py --root data/storage/llm_reward --output-dir data/storage/summary

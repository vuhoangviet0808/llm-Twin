"""Offline-designed reward candidates for the UAV secrecy experiment.

The functions in this module are deterministic helpers only; no LLM or other
external service is used during training.
"""

import math

import numpy as np

SECRECY_GATE_OFFSET = 1.0
RATIO_BASE = 1.0

LLM_REWARD_CONFIGS = {
    "llm_balanced": {
        "secrecy_violation_weight": 2.0,
        "power_violation_weight": 1.0,
        "energy_weight": 0.10,
    },
    "llm_gated": {
        "secrecy_violation_weight": 2.0,
        "power_violation_weight": 1.0,
        "energy_weight": 0.20,
    },
    "llm_ratio": {
        "secrecy_violation_weight": 2.0,
        "power_violation_weight": 1.0,
        "energy_denominator_weight": 0.25,
    },
}

LLM_REWARD_METADATA = {
    "llm_balanced": {
        "designer": "LLM-assisted offline reward design",
        "purpose": "Additive secrecy and normalized-energy objective with invalid-state penalties.",
        "formula": "tanh(positive_ssr - 2*secrecy_violation - power_violation - 0.10*energy_norm)",
    },
    "llm_gated": {
        "designer": "LLM-assisted offline reward design",
        "purpose": "Increase the energy objective as secrecy performance improves.",
        "formula": "tanh(positive_ssr - 2*secrecy_violation - power_violation - 0.20*secrecy_gate*energy_norm)",
    },
    "llm_ratio": {
        "designer": "LLM-assisted offline reward design",
        "purpose": "Use a numerically safe secrecy-energy surrogate without physical-energy division.",
        "formula": "tanh(positive_ssr/(1 + 0.25*energy_norm) - 2*secrecy_violation - power_violation)",
    },
}


def normalized_energy(energy_raw, energy_min, energy_max):
    """Normalize with the repository convention and safely clip to [0, 1]."""
    denominator = float(energy_max - energy_min)
    if denominator == 0.0:
        return 0.0
    value = (float(energy_raw) - float(energy_min)) / denominator
    return float(np.clip(value, 0.0, 1.0))


def extract_reward_components(env, movement_speed, energy_function,
                               energy_min, energy_max):
    """Extract reward terms after ``env.update_channel_capacity()``."""
    raw_secrecy = []
    for user in env.user_list:
        raw_secrecy.append(
            float(user.capacity - np.max(env.eavesdrop_capacity_array[:, user.index]))
        )

    positive_ssr = float(sum(max(0.0, value) for value in raw_secrecy))
    secrecy_violation = float(sum(max(0.0, -value) for value in raw_secrecy))
    power = abs(np.trace(env.UAV.G * env.UAV.G.H))
    power_max = abs(env.UAV.G_Pmax)
    power_violation = float(max(0.0, (power - power_max) / env.power_factor))
    energy_raw = float(energy_function(movement_speed))
    energy_norm = normalized_energy(energy_raw, energy_min, energy_max)
    return {
        "positive_ssr": positive_ssr,
        "secrecy_violation": secrecy_violation,
        "power_violation": power_violation,
        "energy_raw": energy_raw,
        "energy_norm": energy_norm,
    }


def compute_llm_reward(reward_name, components):
    """Return ``(raw_reward, normal_reward)`` for one registered candidate."""
    if reward_name not in LLM_REWARD_CONFIGS:
        raise ValueError("unknown LLM reward: {}".format(reward_name))
    config = LLM_REWARD_CONFIGS[reward_name]
    positive_ssr = components["positive_ssr"]
    violation = components["secrecy_violation"]
    power_violation = components["power_violation"]
    energy_norm = components["energy_norm"]

    if reward_name == "llm_balanced":
        raw_reward = (positive_ssr
                      - config["secrecy_violation_weight"] * violation
                      - config["power_violation_weight"] * power_violation
                      - config["energy_weight"] * energy_norm)
    elif reward_name == "llm_gated":
        secrecy_gate = positive_ssr / (positive_ssr + SECRECY_GATE_OFFSET)
        raw_reward = (positive_ssr
                      - config["secrecy_violation_weight"] * violation
                      - config["power_violation_weight"] * power_violation
                      - config["energy_weight"] * secrecy_gate * energy_norm)
    else:
        secrecy_energy_term = positive_ssr / (
            RATIO_BASE + config["energy_denominator_weight"] * energy_norm)
        raw_reward = (secrecy_energy_term
                      - config["secrecy_violation_weight"] * violation
                      - config["power_violation_weight"] * power_violation)

    raw_reward = float(raw_reward)
    return raw_reward, float(math.tanh(raw_reward))

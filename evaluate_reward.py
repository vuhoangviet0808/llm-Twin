"""Deterministic evaluation for a saved Twin-TD3/DDPG experiment."""

import argparse
import csv
import json
import math
import os
import random

import numpy as np
import torch

from env import MiniSystem, P_i, P_0, U2_tip, s, d_0, p, A, m, g
from td3 import Agent as TD3Agent
from ddpg import Agent as DDPGAgent


REWARD_NAMES = ['ssr', 'see', 'llm_balanced', 'llm_gated', 'llm_ratio']


def metric_energy(v_t):
    """Match load_and_plot.py: displacement is converted to speed first."""
    delta_time = 0.1
    v_0 = (m * g / (A * 2 * p)) ** 0.5
    energy_1 = P_0 + 3 * P_0 * abs(v_t) ** 2 / U2_tip + 0.5 * d_0 * p * s * A * abs(v_t) ** 3
    energy_2 = P_i * ((
        (1 + abs(v_t) ** 4 / (4 * v_0 ** 4)) ** 0.5
        - abs(v_t) ** 2 / (2 * v_0 ** 2)
    ) ** 0.5)
    return delta_time * (energy_1 + energy_2)


def deterministic_action(agent, observation):
    agent.actor.eval()
    with torch.no_grad():
        state = torch.tensor(observation, dtype=torch.float32, device=agent.actor.device)
        action = agent.actor(state)
    agent.actor.train()
    return action.cpu().numpy()


def build_agent(agent_class, system, agent_name, input_dims, n_actions,
                layer_sizes, learning_rate=(0.0001, 0.001)):
    return agent_class(
        alpha=learning_rate[0], beta=learning_rate[1], input_dims=[input_dims],
        tau=0.001, env=system, batch_size=64, layer1_size=layer_sizes[0],
        layer2_size=layer_sizes[1], layer3_size=layer_sizes[2],
        layer4_size=layer_sizes[3], n_actions=n_actions, max_size=1,
        agent_name=agent_name)


def evaluate(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(args.model_path)
    if args.reward not in REWARD_NAMES:
        raise ValueError('unsupported reward: {}'.format(args.reward))

    system = MiniSystem(
        user_num=2, RIS_ant_num=4, UAV_ant_num=4, if_dir_link=1,
        if_with_RIS=True, if_move_users=True, if_movements=True,
        reverse_x_y=(False, False), if_UAV_pos_state=True,
        reward_design=args.reward, project_name='evaluation/{}_{}'.format(
            os.path.basename(os.path.normpath(args.model_path)), args.seed), step_num=100)
    agent_class = TD3Agent if args.drl == 'td3' else DDPGAgent
    agent_1 = build_agent(agent_class, system, 'G_and_Phi', system.get_system_state_dim(),
                          system.get_system_action_dim() - 2, (800, 600, 512, 256))
    agent_2 = build_agent(agent_class, system, 'UAV', 3, 2, (400, 300, 256, 128))

    if args.drl == 'td3':
        agent_1.load_models(
            load_file_actor=os.path.join(args.model_path, 'Actor_G_and_Phi_TD3'),
            load_file_critic_1=os.path.join(args.model_path, 'Critic_1_G_and_Phi_TD3'),
            load_file_critic_2=os.path.join(args.model_path, 'Critic_2_G_and_Phi_TD3'))
        agent_2.load_models(
            load_file_actor=os.path.join(args.model_path, 'Actor_UAV_TD3'),
            load_file_critic_1=os.path.join(args.model_path, 'Critic_1_UAV_TD3'),
            load_file_critic_2=os.path.join(args.model_path, 'Critic_2_UAV_TD3'))
    else:
        agent_1.load_models(
            load_file_actor=os.path.join(args.model_path, 'Actor_G_and_Phi_ddpg'),
            load_file_critic=os.path.join(args.model_path, 'Critic_G_and_Phi_ddpg'))
        agent_2.load_models(
            load_file_actor=os.path.join(args.model_path, 'Actor_UAV_ddpg'),
            load_file_critic=os.path.join(args.model_path, 'Critic_UAV_ddpg'))

    rows = []
    for episode in range(args.episodes):
        system.reset()
        observation_1 = system.observe()
        observation_2 = list(system.UAV.coordinate)
        ssr_values, see_values = [], []
        total_energy = 0.0
        terminated_early = False
        for step in range(system.step_num):
            old_position = np.array(system.UAV.coordinate, dtype=float)
            action_1 = deterministic_action(agent_1, observation_1)
            action_2 = deterministic_action(agent_2, observation_2)
            new_state, reward, done, info = system.step(
                action_0=action_2[0], action_1=action_2[1],
                G=action_1[:2 * system.UAV.ant_num * system.user_num],
                Phi=action_1[2 * system.UAV.ant_num * system.user_num:],
                set_pos_x=action_2[0], set_pos_y=action_2[1])
            displacement = float(np.linalg.norm(np.array(system.UAV.coordinate) - old_position))
            energy = metric_energy(displacement / 0.1)
            ssr = float(info['positive_ssr'])
            ssr_values.append(ssr)
            see_values.append(ssr / energy if energy != 0 else 0.0)
            total_energy += energy
            observation_1, observation_2 = new_state, list(system.UAV.coordinate)
            if done:
                terminated_early = True
                break
        rows.append({
            'episode': episode,
            'average_ssr': float(np.mean(ssr_values)) if ssr_values else 0.0,
            'total_energy_kj': total_energy / 1000.0,
            'average_see': float(np.mean(see_values)) if see_values else 0.0,
            'episode_length': len(ssr_values),
            'boundary_violation': terminated_early,
        })
        print('episode {}: average SSR {:.6f}, total UAV energy {:.6f} kJ, average SEE {:.6f}, length {}, boundary violation {}'.format(
            episode, rows[-1]['average_ssr'], rows[-1]['total_energy_kj'],
            rows[-1]['average_see'], rows[-1]['episode_length'], terminated_early))

    output_dir = args.output_dir or os.path.join(args.model_path, 'evaluation')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation_metrics.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'model_path': os.path.abspath(args.model_path), 'drl': args.drl,
        'reward': args.reward, 'seed': args.seed, 'episodes': args.episodes,
        'mean_ssr': float(np.mean([r['average_ssr'] for r in rows])),
        'std_ssr': float(np.std([r['average_ssr'] for r in rows])),
        'mean_energy_kj': float(np.mean([r['total_energy_kj'] for r in rows])),
        'std_energy_kj': float(np.std([r['total_energy_kj'] for r in rows])),
        'mean_see': float(np.mean([r['average_see'] for r in rows])),
        'std_see': float(np.std([r['average_see'] for r in rows])),
    }
    with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as handle:
        json.dump({'summary': summary, 'episodes': rows}, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--drl', choices=['td3', 'ddpg'], default='td3')
    parser.add_argument('--reward', choices=REWARD_NAMES, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--output-dir', default=None)
    evaluate(parser.parse_args())

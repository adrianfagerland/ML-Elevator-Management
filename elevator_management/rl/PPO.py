# PPO-LSTM

import time
from typing import Type

import gymnasium as gym
import ml.api  # needs to be imported for the env registration. Can this be moved to __init__.py?
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.utils.env_checker import check_env
from rl.network import (
    OUT_HIDDEN_SIZE,
    PRE_HIDDEN_SIZE,
    ElevatorNetwork,
    alphaLSTMNetwork,
)
from vis.console import ConsoleVisualizer
import os
from pathlib import Path
import datetime
from torch.utils.tensorboard.writer import SummaryWriter

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 1
T_horizon = 20

"""
This file is based on the PPO-LSTM implementation of MinimalRL.
https://github.com/seungeunrho/minimalRL

The major change being that the structure of the network is not fixed to more easily test different architectures. 
Also made the code more readable (see original code)
"""


class PPO:
    def __init__(self, network_architecture: Type[ElevatorNetwork], env: gym.Env, log_folder=Path("./logs")):
        super().__init__()
        # store the enviromnent to train the network on
        self.env = env

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.log_folder = log_folder / current_time
        os.makedirs(self.log_folder, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_folder)

        self.num_floors = env.get_wrapper_attr("num_floors")

        # initiliaze the network on which to train
        self.model = network_architecture(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            num_floors=self.num_floors,
        )

        self.data = []

        self.opt = optim.Adam(self.model.parameters())

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        extracted_state_list, extracted_state_prime_list = [], []
        action_list, reward_list, prob_action_list = [], [], []

        hidden_inf_in_list, hidden_inf_out_list, done_list = [], [], []

        for transition in self.data:
            fs, a, r, fs_prime, prob_a, h_in, h_out, done = transition

            extracted_state_list.append(fs)
            action_list.append([a])
            reward_list.append([r])
            extracted_state_prime_list.append(fs_prime)
            prob_action_list.append([prob_a])
            hidden_inf_in_list.append(h_in)
            hidden_inf_out_list.append(h_out)
            done_mask = 0 if done else 1
            done_list.append(done_mask)

        done_mask = th.tensor(done_list, dtype=th.float)
        r = th.tensor(reward_list).squeeze(1)
        log_prob_a = th.tensor(prob_action_list)
        self.data = []
        a = action_list
        fs = th.stack(extracted_state_list)
        fs.requires_grad = True
        fs_prime = th.stack(extracted_state_prime_list)
        fs_prime.requires_grad = True
        return fs, a, r, fs_prime, done_mask, log_prob_a, hidden_inf_in_list[0], hidden_inf_out_list[0]

    def update_parameters(self):
        states, actions, reward, states_prime, done_mask, log_prob_a, hidden_in_0, hidden_out_0 = self.make_batch()

        # (h1_in, h2_in), (h1_out, h2_out) = h_in_0, h_out_0
        # detach? first hidden state
        hidden_states_0 = []
        for ele_hidden_in_0 in hidden_in_0:
            (pre_h, pre_c), (comm_h, comm_c) = ele_hidden_in_0
            hidden_states_0.append(((pre_h.detach(), pre_c.detach()), (comm_h.detach(), comm_c.detach())))
        # detach? first hidden state
        hidden_states_1 = []
        for ele_hidden_in_0 in hidden_out_0:
            (pre_h, pre_c), (_, _) = ele_hidden_in_0
            hidden_states_1.append(((pre_h.detach(), pre_c.detach()), 0))

        for i in range(K_epoch):
            v_prime, _ = self.model.forward_critic(states_prime, hidden_states_1)

            td_target = reward + gamma * v_prime * done_mask
            v_s, _ = self.model.forward_critic(states, hidden_states_0)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = th.tensor(advantage_lst, dtype=th.float)

            pi, _ = self.model.forward_actor(states, hidden_states_0)

            log_prob_a_prime = self.model.get_log_prob(pi, actions)

            ratio = th.exp(log_prob_a_prime - log_prob_a)  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -th.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.opt.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.opt.step()

    def calculate_density(self, n_episode: int, total_episodes: int) -> float:
        MIN_DENSITY = 0.2
        MAX_DENSITY = 3
        return MIN_DENSITY + n_episode / total_episodes * (MAX_DENSITY - MIN_DENSITY)

    def train(self, episode_length=40_000, print_interval=100, save_interval=None):
        score = 0.0
        print_interval = print_interval
        print("Start training!", flush=True)

        start_time = time.time()
        num_steps = 0
        tmp_num_steps = 0
        for n_epi in range(episode_length):
            # Reset with increasing density
            obs, info = self.env.reset(options={"density": self.calculate_density(n_epi, episode_length)})
            last_print_time = time.time()

            done = False
            num_elevators = obs["num_elevators"][0]
            hidden_inf_out = [self.model._generate_empty_hidden_state() for _ in range(num_elevators)]

            while not done:
                for t in range(T_horizon):
                    hidden_inf_in = hidden_inf_out
                    extracted_features = self.model.extract_features(obs)
                    prob, hidden_inf_out = self.model.forward_actor(extracted_features, hidden_inf_in)

                    action, log_prob_a = self.model.sample_action_from_output(prob)
                    obs_prime, reward, done, truncated, info = self.env.step(action)
                    # visualizer.visualize(s_prime, a)
                    # time.sleep(1)
                    num_steps += 1
                    tmp_num_steps += 1

                    extraced_features_prime = self.model.extract_features(obs_prime)
                    # convert r to float (otherwise ide doesnt understand)
                    reward = float(reward)

                    self.put_data(
                        (
                            extracted_features,
                            action,
                            reward,
                            extraced_features_prime,
                            log_prob_a,
                            hidden_inf_in,
                            hidden_inf_out,
                            done,
                        )
                    )
                    obs = obs_prime

                    score += reward
                    if done:
                        break

                self.update_parameters()

                if tmp_num_steps > print_interval:
                    tmp_num_steps -= print_interval
                    time_since_start = time.time() - start_time
                    time_since_last_print = time.time() - last_print_time
                    percentage_done = (
                        (info["num_people_arrived"] + info["num_walked_stairs"]) / info["total_arrivals"] * 100
                    )

                    print(
                        f"#epoch {n_epi} #steps performed:{num_steps} #steps_per_second {print_interval / time_since_last_print:.3f} last_print {time_since_last_print:.3f} done in % {percentage_done:.3f} training_time {time_since_start:.3f}",
                        flush=True,
                    )
                    last_print_time = time.time()

                    self.writer.add_scalar("Steps/Episode", num_steps, n_epi)
                    self.writer.add_scalar("Steps/SPS", num_steps, print_interval / time_since_last_print)
                    self.writer.add_scalar("Steps/Training_Time", num_steps, time_since_start)
                    self.writer.add_scalar("Steps/Episode_done", num_steps, percentage_done)

            print(
                "# of episode: {}, avg score: {:.3f} time_since_start: {}".format(
                    n_epi,
                    score,
                    round(time.time() - start_time, 2),
                ),
                flush=True,
            )

            self.writer.add_scalar("Train/Episode_Reward", n_epi, score)
            self.writer.add_scalar("Train/Episode_Walked", n_epi, info["num_walked_stairs"])
            self.writer.add_scalar("Train/Episode_Arrived", n_epi, info["num_people_arrived"])
            self.writer.add_scalar("Train/Episode_Steps", n_epi, num_steps)
            score = 0.0

        self.env.close()

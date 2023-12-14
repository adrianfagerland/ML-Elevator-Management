# PPO-LSTM

from typing import Type

import gymnasium as gym
import ml.api  # needs to be imported for the env registration

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

from vis.console import print_elevator

import time

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
    def __init__(self, network_architecture: Type[ElevatorNetwork], env: gym.Env):
        super().__init__()
        # store the enviromnent to train the network on
        self.env = env

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
        (
            fs_lst,
            a_lst,
            r_lst,
            fs_prime_lst,
            prob_a_lst,
            h_in_lst,
            h_out_lst,
            done_lst,
        ) = ([], [], [], [], [], [], [], [])
        for transition in self.data:
            fs, a, r, fs_prime, prob_a, h_in, h_out, done = transition

            fs_lst.append(fs)
            a_lst.append([a])
            r_lst.append([r])
            fs_prime_lst.append(fs_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append(done_mask)

        done_mask = th.tensor(done_lst, dtype=th.float)
        r = th.tensor(r_lst).squeeze(1)
        log_prob_a = th.tensor(prob_a_lst)
        self.data = []
        a = a_lst
        fs = th.stack(fs_lst)
        fs.requires_grad = True
        fs_prime = th.stack(fs_prime_lst)
        fs_prime.requires_grad = True
        return fs, a, r, fs_prime, done_mask, log_prob_a, h_in_lst[0], h_out_lst[0]

    def update_parameters(self):
        s, a, r, s_prime, done_mask, log_prob_a, h_in_0, h_out_0 = self.make_batch()

        # (h1_in, h2_in), (h1_out, h2_out) = h_in_0, h_out_0
        # detach? first hidden state
        hidden_states_0 = []
        for ele_hidden_in_0 in h_in_0:
            (pre_h, pre_c), (comm_h, comm_c) = ele_hidden_in_0
            hidden_states_0.append(
                ((pre_h.detach(), pre_c.detach()), (comm_h.detach(), comm_c.detach()))
            )
        # detach? first hidden state
        hidden_states_1 = []
        for ele_hidden_in_0 in h_out_0:
            (pre_h, pre_c), (_, _) = ele_hidden_in_0
            hidden_states_1.append(((pre_h.detach(), pre_c.detach()), 0))

        for i in range(K_epoch):
            v_prime, _ = self.model.forward_critic(s_prime, hidden_states_1)

            td_target = r + gamma * v_prime * done_mask
            v_s, _ = self.model.forward_critic(s, hidden_states_0)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = th.tensor(advantage_lst, dtype=th.float)

            pi, _ = self.model.forward_actor(s, hidden_states_0)

            log_prob_a_prime = self.model.get_log_prob(pi, a)

            ratio = th.exp(log_prob_a_prime - log_prob_a)  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -th.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.opt.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.opt.step()

    def train(self, episode_length=40_000, save_model=None, save_interval=None):
        score = 0.0
        print_interval = 1
        print("Start training!",flush=True)
        num_steps = 0
        start_time = time.time()
        for n_epi in range(episode_length):
            s, _ = self.env.reset()  # this determines how many elevators for this episode
            #print_elevator(s, setup=True)
            done = False
            num_elevators = s["num_elevators"][0]
            hidden_inf_out = [
                self.model._generate_empty_hidden_state() for _ in range(num_elevators)
            ]
            while not done:
                for t in range(T_horizon):
                    hidden_inf_in = hidden_inf_out
                    fs = self.model.extract_features(s)
                    prob, hidden_inf_out = self.model.forward_actor(fs, hidden_inf_in)

                    a, log_prob_a = self.model.sample_action_from_output(prob)

                    s_prime, r, done, truncated, info = self.env.step(a)
                    
                    num_steps += 1

                    #print_elevator(s_prime, 1, previous_action=a, setup=True)
                    fs_prime = self.model.extract_features(s_prime)
                    # convert r to float (otherwise ide doesnt understand)
                    r = float(r)
                    # get probability of action a

                    self.put_data((fs, a,r / 100.0,fs_prime,log_prob_a,hidden_inf_in,hidden_inf_out,done,))
                    s = s_prime
                    
                    score += r
                    if done:
                        break

                self.update_parameters()
            
            print(f"Num steps performed:{num_steps}", flush=True)
            num_steps = 0
            if n_epi % print_interval == 0:
                print(
                    "# of episode :{}, avg score : {:.1f} time_since_start: {}".format(
                        n_epi, score / print_interval, round(time.time() - start_time,2)
                    ), flush=True
                )
                score = 0.0
            # after training
            if(save_interval is not None and n_epi % save_interval == 0 and n_epi != 0):
                assert isinstance(save_model, str)
                print(f"Save model on episode {n_epi}", flush=True)
                tmp_path = save_model[:-2]
                tmp_path +="_" + str(n_epi) + ".ml"
                th.save(self.model.state_dict(), tmp_path)
        
        # after training
        if(save_model is not None):
            th.save(self.model.state_dict(), save_model)
        
        self.env.close()



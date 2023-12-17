from datetime import datetime

import torch
import numpy as np

import gymnasium as gym

from rl.PPO_v2 import PPO

from time import time
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
################################### Training ###################################
def train():
    print("============================================================================================")



    env_name = "Elevator-v0"
    env = gym.make(env_name, num_floors=10, num_elevators=3, num_arrivals=100, observation_type='discrete', action_type='discrete')
    # check_env(env.unwrapped)


    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    log_freq = 2000           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    #####################################################


    ################ PPO hyperparameters ################
    update_timestep = 1000      # update policy every n timesteps
    K_epochs = 40               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)



    # state space dimension
    state_dim = env.observation_space.shape[0] # type: ignore

    action_dim = sum(env.action_space.nvec) # type: ignore
    action_view = env.action_space.shape + (env.action_space.nvec[0],) # type: ignore 

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    # Create Log Path
    log_dir = Path(os.getcwd())
    # Check if in correct folder (might not be neccessary idk)
    if(os.path.isdir(log_dir/"elevator_management")):
        log_dir = log_dir/"elevator_management"

    log_dir = log_dir/"logs"



    #### get number of log files in log directory
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    log_dir = log_dir/current_time
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print("current logging folder:" + str(log_dir))

    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        #env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, action_view, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # track total training time
    start_time = datetime.datetime.now().replace(microsecond=0)
    start_time_sec = time()
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")


    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0
    averaged_ep_reward = 0
    time_step = 0
    i_episode = 0

    last_time = time()
    # training loop
    while time_step <= max_training_timesteps:

        state, info = env.reset()
        current_ep_reward = 0
        done = False
        while not done:

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)

            reward = float(reward)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # log in logging file
            if time_step % log_freq == 0:
                current_time = time()
                sps = log_freq / (current_time - last_time)
                writer.add_scalar("Steps/SPS", time_step, sps)

                percentage_done = (info['num_people_arrived'] + info["num_walked_stairs"]) / info['total_arrivals'] * 100
                
                print(f"steps {time_step} SPS {sps:.2f} train {current_time - start_time_sec:.2f} done% {percentage_done:.2f}")
                last_time = current_time
            # break; if the episode is over
            if done:
                break
        
        
        averaged_ep_reward = current_ep_reward * 0.08 + 0.92 * averaged_ep_reward
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        print(f"#Epi {i_episode} rew {current_ep_reward:.2f} ave_rew {averaged_ep_reward:.2f} walk {info['num_walked_stairs']} arr {info['num_people_arrived']}")

        writer.add_scalar("Train/Episode_Reward", i_episode, current_ep_reward)
        writer.add_scalar("Train/Episode_Walked",i_episode, info['num_walked_stairs'])
        writer.add_scalar("Train/Episode_Arrived",i_episode, info['num_people_arrived'])
        writer.add_scalar("Train/Episode_Steps", i_episode, time_step)
        writer.add_scalar("Train/Averg_reward", i_episode, averaged_ep_reward)

        i_episode += 1

    writer.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
    
    
    
    
    
    
    
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
from rl.new_network import SimpleMLP, AlphaNetwork, ActorCritic
import os
from pathlib import Path
import datetime
from torch.utils.tensorboard.writer import SummaryWriter
################################### Training ###################################
def train():
    print("============================================================================================")


    ################ General hyperparameters ################
    env_name = "Elevator-v0"
    env = gym.make(env_name, num_floors=10, num_elevators=3, num_arrivals=100, observation_type='discrete', action_type='discrete')
    # check_env(env.unwrapped)
    policy_net = AlphaNetwork

    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

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



    ################ LOG hyperparameters ################
    log_freq = 2000           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(10000)          # save model frequency (in num timesteps)
    avg_reward_update_per = 0.08
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
    # dummy reset to obtain info
    obs, info = env.reset()
    #
    ppo_agent = PPO(policy_net,state_dim, action_dim, action_view, lr_actor, lr_critic, gamma, K_epochs, eps_clip, info)

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
            action = ppo_agent.select_action(state, info)
            state, reward, done, truncated, info = env.step(action)

            reward = float(reward)
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            if time_step % save_model_freq == 0:
                model_path = log_dir/"time_step"
                ppo_agent.save(model_path)

            # update PPO agent
            if time_step % update_timestep == 0:
                log_time_start = time()
                ppo_agent.update()
                writer.add_scalar("Update/time", time_step, time() - log_time_start)
                print(f"update num {time_step // update_timestep} in time {time() - log_time_start:.2f}")

            # log in logging file
            if time_step % log_freq == 0:
                current_time = time()
                sps = log_freq / (current_time - last_time)
                writer.add_scalar("Steps/SPS", time_step, sps)
                writer.add_scalar("Steps/Time", time_step, current_time - last_time)
                print(f"Num of steps {time_step}. With  {sps:.2f} Steps per sec and Total Training time {current_time - start_time_sec:.2f}s")
                last_time = current_time
            # break; if the episode is over
            if done:
                break
        
        if(log_running_episodes < 6):
            averaged_ep_reward = current_ep_reward * 1/2 + 1/2 * averaged_ep_reward
        else:
            averaged_ep_reward = current_ep_reward * avg_reward_update_per + (1-avg_reward_update_per) * averaged_ep_reward
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
    
    
    
    
    
    
    
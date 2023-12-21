from datetime import datetime

import torch
import numpy as np

import gymnasium as gym

from rl.PPO import PPO
import os
# disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import argparse
 
from time import time
import gymnasium as gym
import ml.api  # needs to be imported for the env registration. Can this be moved to __init__.py?
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.utils.env_checker import check_env
from rl.network import SimpleMLP, AlphaNetwork, ActorCritic, BidirectNetwork, AlphaNetwork_output
import os
from pathlib import Path
import datetime
from torch.utils.tensorboard.writer import SummaryWriter
################################### Training ###################################

parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-t", "--training_steps", type=int, default=int(3e7))
parser.add_argument("-m", "--model_name", type=str, default='alpha')
parser.add_argument("--time_date", type=str)
parser.add_argument("--time_hours", type=str)
parser.add_argument("--max_training_timesteps", type=int)
parser.add_argument("--update_timestep", type=int)
parser.add_argument("--K_epochs", type=int)
parser.add_argument("--eps_clip", type=float)
parser.add_argument("--gamma", type=float)
parser.add_argument("--lr_actor", type=float)
parser.add_argument("--lr_critic", type=float)
parser.add_argument("--random_seed", type=int)
parser.add_argument("--log_freq", type=int)
parser.add_argument("--save_model_freq", type=int)
parser.add_argument("--avg_reward_update_per", type=float)
parser.add_argument("--num_arrivals", type=int)
parser.add_argument("--num_elevators_min", type=int)
parser.add_argument("--num_elevators_max", type=int)
parser.add_argument("--num_floors", type=int)
parser.add_argument("--message", type=str)
parser.add_argument("--density", type=float)



# Read arguments from command line
args = parser.parse_args()

possible_models = {'alpha':AlphaNetwork,
                    'mlp':SimpleMLP,
                    'birec':BidirectNetwork,
                    'output':AlphaNetwork_output}

if(args.time_date is None or args.time_hours is None):
    time_date = datetime.datetime.now().strftime("%Y-%m-%d")
    time_hours = datetime.datetime.now().strftime("%H-%M-%S")
else:
    time_date = args.time_date
    time_hours = args.time_hours


################ General hyperparameters ################
model_name = args.model_name
model = possible_models[model_name]
message = "" if args.message is None else args.message
#####################################################


################ PPO hyperparameters ################

max_training_timesteps =    3_000_000 if args.max_training_timesteps is None else args.max_training_timesteps    # break training loop if timeteps > max_training_timesteps
update_timestep =           1000 if args.update_timestep is None else args.update_timestep     # update policy every n timesteps
K_epochs =                  40    if args.K_epochs is None else args.K_epochs            # update policy for K epochs in one PPO update

eps_clip =  0.2 if args.eps_clip is None else args.eps_clip          # clip parameter for PPO
gamma =     0.90   if args.gamma is None else args.gamma          # discount factor

lr_actor =      0.0003  if args.lr_actor is None else args.lr_actor      # learning rate for actor network
lr_critic =     0.001  if args.lr_critic is None else args.lr_critic      # learning rate for critic network
random_seed =   1   if args.random_seed is None else args.random_seed       # set random seed if required (0 = no random seed)
#####################################################



################ LOG hyperparameters ################
log_freq =              2000 if args.log_freq is None else args.log_freq         # log avg reward in the interval (in num timesteps)
save_model_freq =       50_000 if args.save_model_freq is None else args.save_model_freq       # save model frequency (in num timesteps)
avg_reward_update_per = 0.08 if args.avg_reward_update_per is None else args.avg_reward_update_per
#####################################################


################ Environment hyperparameters ################
num_arrivals =      800 if args.num_arrivals is None else args.num_arrivals   
num_elevators_min = 1 if args.num_elevators_min is None else args.num_elevators_min
num_elevators_max = 1 if args.num_elevators_max is None else args.num_elevators_max
num_floors =        20 if args.num_floors is None else args.num_floors
density =           1 if args.density is None else args.density
#####################################################
print("============================================================================================")


################ General hyperparameters ################
env_name = "Elevator-v0"
env = gym.make(env_name, num_floors=num_floors, num_elevators=(num_elevators_min,num_elevators_max), num_arrivals=num_arrivals, observation_type='array', action_type='array')
# check_env(env.unwrapped)
policy_net = model






print("training environment name : " + env_name)



# state space dimension
state_dim = env.observation_space.shape[0] # type: ignore

action_dim = sum(env.action_space.nvec) # type: ignore
action_view = env.action_space.nvec[0] # type: ignore 

###################### logging ######################

#### log files for multiple runs are NOT overwritten

# Create Log Path
log_dir = Path(os.getcwd())
# Check if in correct folder (might not be neccessary idk)
if(os.path.isdir(log_dir/"elevator_management")):
    log_dir = log_dir/"elevator_management"

log_dir = log_dir/"logs"


#### get number of log files in log directory
current_day = time_date
current_time = time_hours

log_dir = log_dir/current_day/current_time


os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
print("current logging folder:" + str(log_dir))

writer.add_text("model_type", message)
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
obs, info = env.reset(options={'density':density})

ppo_agent = PPO(policy_net,state_dim, action_dim, action_view, lr_actor, lr_critic, gamma, K_epochs, eps_clip, info)

# track total training time
start_time = datetime.datetime.now().replace(microsecond=0)
start_time_sec = time()
print("Started training at (GMT) : ", start_time, flush=True)
print("============================================================================================")


# printing and logging variables

log_running_reward = 0
log_running_episodes = 0
log_update_time = 0

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
        # only one action is provided
        state, reward, done, truncated, info = env.step(action)

        reward = float(reward)
        # saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        time_step +=1
        current_ep_reward += reward

        if time_step % save_model_freq == 0:
            model_path = log_dir/("checkpoint_model-" + str(time_step) + ".ml")
            #writer.add_graph(ppo_agent.policy_old)
            ppo_agent.save(model_path)

        # update PPO agent
        if time_step % update_timestep == 0:
            log_update_start = time()
            ppo_agent.update(buffer_length = update_timestep)
            log_update_end = time()
            writer.add_scalar("Update/time", log_update_end - log_update_start, time_step)

            print(f"Number of Update: {time_step // update_timestep} (Time to update: {log_update_end - log_update_start:.2f})")
            log_update_time += log_update_end - log_update_start


        # log in logging file
        if time_step % log_freq == 0:
            current_time = time()
            # time that was not spend updating
            run_time = current_time - last_time - log_update_time
            # reset time spend updating
            log_update_time = 0
            # steps per second
            sps = log_freq / (run_time)
            writer.add_scalar("Steps/SPS", sps, time_step)
            writer.add_scalar("Steps/Time", run_time, time_step)
            print(f"Total number of steps: {time_step}. The total training time is {run_time:.2f}s (since last call: {sps:.2f} steps/second)", flush=True)
            last_time = current_time
        # break; if the episode is over
        if done:
            break

    # update PPO agent
    log_update_start = time()
    ppo_agent.update(buffer_length = update_timestep)
    log_update_end = time()
    writer.add_scalar("Update/time", log_update_end - log_update_start, time_step)

    print(f"Number of Update: {time_step / update_timestep:.2f} (Time to update: {log_update_end - log_update_start:.2f})")
    log_update_time += log_update_end - log_update_start


    if(log_running_episodes == 0):
        averaged_ep_reward = current_ep_reward
    else:
        averaged_ep_reward = current_ep_reward * avg_reward_update_per + (1-avg_reward_update_per) * averaged_ep_reward
    log_running_reward += current_ep_reward
    log_running_episodes += 1

    print(f"#Episode Number {i_episode} #Elevators {info['num_elevators']}. Steps of last Episode: {info['num_steps']}")
    print(f"Total Episode Reward: {current_ep_reward:.2f} Averaged Reward: {averaged_ep_reward:.2f}. Number of people arrived: {info['num_people_arrived']}. Average Arrival Time: {info['average_arrival_time']:.2f}s")
    
    writer.add_scalar("Train/Num_People_Arrived", info['num_people_arrived'], i_episode)
    writer.add_scalar("Train/Episode_Steps", time_step, i_episode)
    writer.add_scalar("Train/Episode_Reward", current_ep_reward, i_episode)
    writer.add_scalar("Train/Time_to_arrive", info['average_arrival_time'], i_episode)
    num_elevators = info['num_elevators']

    writer.add_scalar(f"Train/Num_People_Arrived:{num_elevators}", info['num_people_arrived'], i_episode)
    writer.add_scalar(f"Train/Episode_Steps:{num_elevators}", time_step, i_episode)
    writer.add_scalar(f"Train/Episode_Reward:{num_elevators}", current_ep_reward, i_episode)
    writer.add_scalar(f"Train/Time_to_arrive:{num_elevators}", info['average_arrival_time'], i_episode)
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


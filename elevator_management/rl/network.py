import torch
import torch.nn as nn
from torch.distributions import Categorical
from rl.alpha_lstm import script_alpha_lstm


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_view, options={}):
        super(ActorCritic, self).__init__()
        self.action_view = action_view
        self.actor_softmax = nn.Softmax(dim=-1)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.options = options
    
    def act(self, state, info):
        
        action_probs = self.forward_actor(state)
        # if in batch mode, change action_view
        in_batch_mode = (len(state.shape) > 1)
        num_elevators = action_probs.shape[-1] // self.action_view

        if in_batch_mode:
            # batch mode
            view = (-1,) + (num_elevators, self.action_view)
        else:
            # in not batch mode 
            # non batch mode
            view = (num_elevators, self.action_view)
        
        action_probs = action_probs.view(view)

        action_probs = self.actor_softmax(action_probs)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_val = self.forward_critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.forward_actor(state)
        # if in batch mode, change action_view
        in_batch_mode = (len(state.shape) > 1)
        num_elevators = action_probs.shape[-1] // self.action_view

        if in_batch_mode:
            # batch mode
            view = (-1,) + (num_elevators, self.action_view)
        else:
            action_probs = action_probs[1:]
            # non batch mode
            view = (num_elevators, self.action_view)
        
        action_probs = action_probs.view(view)


        action_probs = self.actor_softmax(action_probs)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy()
        state_values = self.forward_critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def actor_parameters(self):
        raise NotImplementedError("Needs to be implemeted")
    
    def critic_parameters(self):
        raise NotImplementedError("Needs to be implemeted")

    def forward_actor(self, state):
        raise NotImplementedError("Needs to be implemeted")
    
    def forward_critic(self, state):
        raise NotImplementedError("Needs to be implemeted")
    

class SimpleMLP(ActorCritic):
    def __init__(self, state_dim, action_dim, action_view, options = {}):
        super().__init__(state_dim, action_dim, action_view, options)
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim)
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def actor_parameters(self):
        return self.actor.parameters()
    def critic_parameters(self):
        return self.critic.parameters()
    
    def forward_actor(self, state):
        return self.actor(state)
    
    def forward_critic(self, state):
        return self.critic(state)


class AlphaNetwork(ActorCritic):
    def __init__(self, state_dim, action_dim, action_view, options = {}):
        super().__init__(state_dim, action_dim, action_view, options)
        # actor
        self.group_info_len = options['group_info_len']
        self.elevator_info_len = options['elevator_info_len']
        self.max_elevators = options['max_elevators']
        self.input_size = self.group_info_len + self.elevator_info_len
        self.elevator_action_dim = int(action_dim / self.max_elevators)
        # network params
        self.num_rounds = 3
        self.dropoff = 0.5
        self.comm_size_in = 64
        self.comm_size_out = 32
        self.comm_layers = 1

        self.preprocess_actor = nn.Sequential(nn.Linear(self.input_size, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.comm_size_in),
                                        nn.Tanh())
        self.preprocess_critic = nn.Sequential(nn.Linear(self.input_size, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.comm_size_in),
                                        nn.Tanh())
        self.comm_actor = script_alpha_lstm(self.comm_size_in, self.comm_size_out, self.comm_layers)
        self.comm_critic = script_alpha_lstm(self.comm_size_in, self.comm_size_out, self.comm_layers)

        self.post_actor = nn.Sequential(nn.Linear(self.comm_size_out * self.comm_layers + self.comm_size_in, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.elevator_action_dim))
        
        self.post_critic = nn.Sequential(nn.Linear(self.comm_size_out * self.comm_layers + self.comm_size_in, 64),
                                         nn.LeakyReLU(0.2),
                                         nn.Linear(64,1))


    def split_features(self, features: torch.Tensor):
        # if batch mode then call function on each batch individually
        if(features.dim() > 1):
            output = []
            for sub_features in features[:]:
                output.append(self.split_features(sub_features))
            return torch.stack(output)
        num_elevators = int(features[:self.max_elevators].sum(dim=0).item())
        
        data = features[self.max_elevators:]

        group_info = data[: self.group_info_len]

        output = []
        for ele_idx in range(num_elevators):
            split_features = torch.zeros(self.group_info_len + self.elevator_info_len)
            split_features[: self.group_info_len] = group_info
            feature_tensor = data[
                self.group_info_len
                + ele_idx * self.elevator_info_len : self.group_info_len
                + (ele_idx + 1) * self.elevator_info_len
            ]

            split_features[self.group_info_len :] = feature_tensor
            output.append(split_features)
        
        return torch.stack(output)

    def forward_actor(self, state):
        
        in_batch_mode = state.dim() > 1
        # make state to batch single
        if not in_batch_mode:
            state = state.unsqueeze(0)
        split_features = self.split_features(state)



        batch_size = split_features.shape[0]
        num_elevators = split_features.shape[1]
        # preprocess
        split_features = self.preprocess_actor(split_features)

        comm_hidden = (
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
        )
        alpha = 1
        for _ in range(self.num_rounds):
            final_output, comm_hidden = self.comm_actor(split_features, comm_hidden, alpha)
            alpha *= self.dropoff
        
        comm_h, comm_c = comm_hidden
        # make comm_c into flatten information (i.e. all layers into singel tensor)
        decision_features = comm_c.swapaxes(0,1).flatten(start_dim=1,end_dim=2)
        # copy decision features for all elevators
        decision_features = decision_features.unsqueeze(dim=1).repeat(1, num_elevators, 1)
        # combine features with decision
        split_features = torch.concatenate((split_features,decision_features), dim=2)

        output = self.post_actor(split_features).flatten(start_dim=1, end_dim=2)
        
        if in_batch_mode:
            return output
        return output[0]
    
    def forward_critic(self, state):
        in_batch_mode = state.dim() > 1
        # make state to batch single
        if not in_batch_mode:
            state = state.unsqueeze(0)
        split_features = self.split_features(state)
        
        batch_size = split_features.shape[0]
        num_elevators = split_features.shape[1]
        # preprocess
        split_features = self.preprocess_critic(split_features)

        comm_hidden = (
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
        )
        alpha = 1
        for _ in range(self.num_rounds):
            final_output, comm_hidden = self.comm_critic(split_features, comm_hidden, alpha)
            alpha *= self.dropoff
        
        comm_h, comm_c = comm_hidden
        # make comm_c into flatten information (i.e. all layers into singel tensor)
        decision_features = comm_c.swapaxes(0,1).flatten(start_dim=1,end_dim=2)
        # copy decision features for all elevators
        decision_features = decision_features.unsqueeze(dim=1).repeat(1, num_elevators, 1)
        # combine features with decision
        split_features = torch.concatenate((split_features,decision_features), dim=2)
        
        output = self.post_critic(split_features).flatten(start_dim=1, end_dim=2).sum(dim=1)
        
        if in_batch_mode:
            return output
        return output[0]
    
    def actor_parameters(self):
        return list(self.preprocess_actor.parameters()) + list(self.comm_actor.parameters()) + list(self.post_actor.parameters())
    def critic_parameters(self):
        return list(self.preprocess_critic.parameters()) + list(self.comm_critic.parameters()) + list(self.post_critic.parameters())
    

class AlphaNetwork_output(ActorCritic):
    # sends the output of the alpha lstm instead of the hidden
    def __init__(self, state_dim, action_dim, action_view, options = {}):
        super().__init__(state_dim, action_dim, action_view, options)
        # actor
        self.group_info_len = options['group_info_len']
        self.elevator_info_len = options['elevator_info_len']
        self.max_elevators = options['max_elevators']
        self.input_size = self.group_info_len + self.elevator_info_len
        self.elevator_action_dim = int(action_dim / self.max_elevators)
        # network params
        self.num_rounds = 3
        self.dropoff = 0.5
        self.comm_size_in = 64
        self.comm_size_out = 32
        self.comm_layers = 1

        self.preprocess_actor = nn.Sequential(nn.Linear(self.input_size, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.comm_size_in),
                                        nn.Tanh())
        self.preprocess_critic = nn.Sequential(nn.Linear(self.input_size, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.comm_size_in),
                                        nn.Tanh())
        self.comm_actor = script_alpha_lstm(self.comm_size_in, self.comm_size_out, self.comm_layers)
        self.comm_critic = script_alpha_lstm(self.comm_size_in, self.comm_size_out, self.comm_layers)

        self.post_actor = nn.Sequential(nn.Linear(self.comm_size_out * self.comm_layers + self.comm_size_in, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.elevator_action_dim))
        
        self.post_critic = nn.Sequential(nn.Linear(self.comm_size_out * self.comm_layers + self.comm_size_in, 64),
                                         nn.LeakyReLU(0.2),
                                         nn.Linear(64,1))


    def split_features(self, features: torch.Tensor):
        # if batch mode then call function on each batch individually
        if(features.dim() > 1):
            output = []
            for sub_features in features[:]:
                output.append(self.split_features(sub_features))
            return torch.stack(output)
        num_elevators = int(features[:self.max_elevators].sum(dim=0).item())
        
        data = features[self.max_elevators:]

        group_info = data[: self.group_info_len]

        output = []
        for ele_idx in range(num_elevators):
            split_features = torch.zeros(self.group_info_len + self.elevator_info_len)
            split_features[: self.group_info_len] = group_info
            feature_tensor = data[
                self.group_info_len
                + ele_idx * self.elevator_info_len : self.group_info_len
                + (ele_idx + 1) * self.elevator_info_len
            ]

            split_features[self.group_info_len :] = feature_tensor
            output.append(split_features)
        
        return torch.stack(output)

    def forward_actor(self, state):
        in_batch_mode = state.dim() > 1
        # make state to batch single
        if not in_batch_mode:
            state = state.unsqueeze(0)
        split_features = self.split_features(state)
        
        batch_size = split_features.shape[0]
        num_elevators = split_features.shape[1]
        # preprocess
        split_features = self.preprocess_actor(split_features)

        comm_hidden = (
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
        )
        alpha = 1
        for _ in range(self.num_rounds):
            final_output, comm_hidden = self.comm_actor(split_features, comm_hidden, alpha)
            alpha *= self.dropoff
        
        comm_h, comm_c = comm_hidden
        # make comm_c into flatten information (i.e. all layers into singel tensor)
        #decision_features = comm_c.swapaxes(0,1).flatten(start_dim=1,end_dim=2)
        # copy decision features for all elevators
        #decision_features = decision_features.unsqueeze(dim=1).repeat(1, num_elevators, 1)
        # combine features with decision
        split_features = torch.concatenate((split_features, final_output), dim=2) # type: ignore
        #split_features = torch.concatenate((split_features,decision_features), dim=2)

        output = self.post_actor(split_features).flatten(start_dim=1, end_dim=2)
        
        if in_batch_mode:
            return output
        return output[0]
    
    def forward_critic(self, state):
        in_batch_mode = state.dim() > 1
        # make state to batch single
        if not in_batch_mode:
            state = state.unsqueeze(0)
        split_features = self.split_features(state)
        
        batch_size = split_features.shape[0]
        num_elevators = split_features.shape[1]
        # preprocess
        split_features = self.preprocess_critic(split_features)

        comm_hidden = (
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
        )
        alpha = 1
        for _ in range(self.num_rounds):
            final_output, comm_hidden = self.comm_critic(split_features, comm_hidden, alpha)
            alpha *= self.dropoff
        
        comm_h, comm_c = comm_hidden
        # make comm_c into flatten information (i.e. all layers into singel tensor)
        # make comm_c into flatten information (i.e. all layers into singel tensor)
        #decision_features = comm_c.swapaxes(0,1).flatten(start_dim=1,end_dim=2)
        # copy decision features for all elevators
        #decision_features = decision_features.unsqueeze(dim=1).repeat(1, num_elevators, 1)
        # combine features with decision
        split_features = torch.concatenate((split_features, final_output), dim=2) # type: ignore
        
        output = self.post_critic(split_features).flatten(start_dim=1, end_dim=2).sum(dim=1)
        
        if in_batch_mode:
            return output
        return output[0]
    
    
    def actor_parameters(self):
        return list(self.preprocess_actor.parameters()) + list(self.comm_actor.parameters()) + list(self.post_actor.parameters())
    def critic_parameters(self):
        return list(self.preprocess_critic.parameters()) + list(self.comm_critic.parameters()) + list(self.post_critic.parameters())
    



    


    

class BidirectNetwork(ActorCritic):
    def __init__(self, state_dim, action_dim, action_view, options = {}):
        super().__init__(state_dim, action_dim, action_view, options)
        # actor
        self.group_info_len = options['group_info_len']
        self.elevator_info_len = options['elevator_info_len']
        self.max_elevators = options['max_elevators']
        self.input_size = self.group_info_len + self.elevator_info_len
        self.elevator_action_dim = int(action_dim / self.max_elevators)
        # network params
        self.num_rounds = 3
        self.dropoff = 0.5
        self.comm_size_in = 64
        self.comm_size_out = 32
        self.comm_layers = 2

        self.preprocess_actor = nn.Sequential(nn.Linear(self.input_size, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.comm_size_in),
                                        nn.Tanh())
        self.preprocess_critic = nn.Sequential(nn.Linear(self.input_size, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.comm_size_in),
                                        nn.Tanh())
        self.comm_actor = script_alpha_lstm(self.comm_size_in, self.comm_size_out, self.comm_layers)
        self.comm_critic = script_alpha_lstm(self.comm_size_in, self.comm_size_out, self.comm_layers)

        self.post_actor = nn.Sequential(nn.Linear(self.comm_size_out * self.comm_layers + self.comm_size_in, 64),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(64,self.elevator_action_dim))
        
        self.post_critic = nn.Sequential(nn.Linear(self.comm_size_out * self.comm_layers + self.comm_size_in, 64),
                                         nn.LeakyReLU(0.2),
                                         nn.Linear(64,1))


    def split_features(self, features: torch.Tensor):
        # if batch mode then call function on each batch individually
        if(features.dim() > 1):
            output = []
            for sub_features in features[:]:
                output.append(self.split_features(sub_features))
            return torch.stack(output)
        num_elevators = int(features[:self.max_elevators].sum(dim=0).item())
        
        data = features[self.max_elevators:]

        group_info = data[: self.group_info_len]

        output = []
        for ele_idx in range(num_elevators):
            split_features = torch.zeros(self.group_info_len + self.elevator_info_len)
            split_features[: self.group_info_len] = group_info
            feature_tensor = data[
                self.group_info_len
                + ele_idx * self.elevator_info_len : self.group_info_len
                + (ele_idx + 1) * self.elevator_info_len
            ]

            split_features[self.group_info_len :] = feature_tensor
            output.append(split_features)
        
        return torch.stack(output)

    def forward_actor(self, state):
        split_features = self.split_features(state)
        # make split features to batch
        if(split_features.dim() < 3):
            split_features = split_features.unsqueeze(0)
        
        batch_size = split_features.shape[0]
        num_elevators = split_features.shape[1]
        # preprocess
        split_features = self.preprocess(split_features)

        comm_hidden = (
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
        )
        alpha = 1
        for _ in range(self.num_rounds):
            final_output, comm_hidden = self.comm_actor(split_features, comm_hidden)
            alpha *= self.dropoff
        
        comm_h, comm_c = comm_hidden
        # make comm_c into flatten information (i.e. all layers into singel tensor)
        decision_features = comm_c.swapaxes(0,1).flatten(start_dim=1,end_dim=2)
        # copy decision features for all elevators
        decision_features = decision_features.unsqueeze(dim=1).repeat(1, num_elevators, 1)
        # combine features with decision
        split_features = torch.concatenate((split_features,decision_features), dim=2)
        return self.post_actor(split_features).flatten(start_dim=1, end_dim=2)
    
    def forward_critic(self, state):
        split_features = self.split_features(state)
        # make split features to batch
        if(split_features.dim() < 3):
            split_features = split_features.unsqueeze(0)
        
        batch_size = split_features.shape[0]
        num_elevators = split_features.shape[1]
        # preprocess
        split_features = self.preprocess(split_features)

        comm_hidden = (
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
            torch.zeros([self.comm_layers, batch_size, self.comm_size_out], dtype=torch.float),
        )
        alpha = 1
        for _ in range(self.num_rounds):
            final_output, comm_hidden = self.comm_critic(split_features, comm_hidden)
            alpha *= self.dropoff
        
        comm_h, comm_c = comm_hidden
        # make comm_c into flatten information (i.e. all layers into singel tensor)
        decision_features = comm_c.swapaxes(0,1).flatten(start_dim=1,end_dim=2)
        # copy decision features for all elevators
        decision_features = decision_features.unsqueeze(dim=1).repeat(1, num_elevators, 1)
        # combine features with decision
        split_features = torch.concatenate((split_features,decision_features), dim=2)
        
        return self.post_critic(split_features).flatten(start_dim=1, end_dim=2).sum(dim=1)
    
    def actor_parameters(self):
        return list(self.preprocess_actor.parameters()) + list(self.comm_actor.parameters()) + list(self.post_actor.parameters())
    def critic_parameters(self):
        return list(self.preprocess_critic.parameters()) + list(self.comm_critic.parameters()) + list(self.post_critic.parameters())
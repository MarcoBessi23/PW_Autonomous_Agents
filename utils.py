'''NeuralNet.py - Defines the neural network architecture for the PPO agent'''
import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical
from einops.layers.torch import Rearrange

class ReplayBuffer:
    '''Buffer memory to store environment interactions'''
    def __init__(self, rollout_size, state_dim):
        self.actions = torch.zeros((rollout_size))
        self.states = torch.zeros((rollout_size, *state_dim))
        self.logprobs = torch.zeros((rollout_size))
        self.rewards = torch.zeros((rollout_size,))
        self.state_values = torch.zeros((rollout_size,))
        self.is_terminals = torch.zeros((rollout_size,), dtype=torch.float32)

def init_layer_weights(layer:nn.Module, std = np.sqrt(2), bias_const = 0.0):
    '''
    Orthogonal initialization of layer weights
    '''
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    '''
    Actor-Critic network for PPO
    '''
    def __init__(self, action_dim, num_channels):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.Net = nn.Sequential(
            init_layer_weights(nn.Conv2d(num_channels, 16, 5, stride=1, padding=2)),
            nn.ReLU(),
            init_layer_weights(nn.Conv2d(16, 32, 5, stride=1, padding=2)),
            nn.ReLU(),
            init_layer_weights(nn.Conv2d(32, 64, 5, stride=1, padding=2)),
            nn.ReLU(),
            init_layer_weights(nn.Conv2d(64, 64, 5, stride=1, padding=2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            Rearrange('b c h w -> b (c h w)'),
            init_layer_weights(nn.Linear(64 * 4 * 4, 1024)),
            nn.ReLU(),
            nn.Dropout(0.1)
            )
        self.actor = nn.Sequential(
            init_layer_weights(nn.Linear(1024, 512)),
            nn.ReLU(),
            init_layer_weights(nn.Linear(512, 256)),
            nn.ReLU(),
            init_layer_weights(nn.Linear(256, action_dim), std=0.01)
        )
        self.critic = nn.Sequential(
            init_layer_weights(nn.Linear(1024, 512)),
            nn.ReLU(),
            init_layer_weights(nn.Linear(512, 256)),
            nn.ReLU(),
            init_layer_weights(nn.Linear(256, 1), std=1.0)
            )

    def get_value(self, state):
        '''
        get state value
        '''
        return self.critic(self.Net(state))

    def act(self, state):
        '''
        return the action to take in the state
        '''
        state = torch.FloatTensor(state)
        logits = self.actor(self.Net(state))
        dist = Categorical(logits = logits)
        action = dist.sample()
        return action.detach(), dist.log_prob(action), self.critic(self.Net(state))

    def evaluate(self, state, action):
        '''
        return log probability, state value and entropy
        of the action taken in the state
        '''
        state = torch.FloatTensor(state)
        logits = self.actor(self.Net(state))
        dist = Categorical(logits = logits)
        return dist.log_prob(action), self.critic(self.Net(state)), dist.entropy()

class ActorCriticSmall(nn.Module):
    '''
    Smaller Actor-Critic network for PPO
    '''
    def __init__(self, action_dim, num_channels=3):
        super(ActorCriticSmall, self).__init__()
        self.action_dim = action_dim
        self.convnet = nn.Sequential(
            init_layer_weights(nn.Conv2d(num_channels, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            init_layer_weights(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            init_layer_weights(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            Rearrange('b c h w -> b (c h w)'),
            init_layer_weights(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
            nn.Dropout(0.1)  # Light dropout for regularization
            )
        self.actor = nn.Sequential(
            init_layer_weights(nn.Linear(512, 256)),
            nn.ReLU(),
            init_layer_weights(nn.Linear(256, action_dim), std=0.01)
        )
        self.critic = nn.Sequential(
            init_layer_weights(nn.Linear(512, 256)),
            nn.ReLU(),
            init_layer_weights(nn.Linear(256, 1), std=1.0)
            )
    def get_value(self, state):
        '''
        get state value
        '''
        return self.critic(self.convnet(state))

    def act(self, state):
        '''
        return the action to take in the state
        '''
        state = torch.FloatTensor(state)
        logits = self.actor(self.convnet(state))
        dist = Categorical(logits = logits)
        action = dist.sample()
        return action.detach(), dist.log_prob(action), self.critic(self.convnet(state))

    def evaluate(self, state, action):
        '''
        return log probability, state value and entropy
        of the action taken in the state
        '''
        state = torch.FloatTensor(state)
        logits = self.actor(self.convnet(state))
        dist = Categorical(logits = logits)
        return dist.log_prob(action), self.critic(self.convnet(state)), dist.entropy()

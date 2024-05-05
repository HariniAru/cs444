import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import DQN
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.action_size = action_size
        
        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # Initialize a target network and initialize the target network to the policy net
        ### CODE ###
        self.target_net = DQN(action_size)
        self.target_net.to(device)
        self.update_target_net()


    def load_policy_net(self, path):
        self.policy_net = torch.load(path)           

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        ### CODE ###
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # pass


    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Randomly select an action, ensure it's a tensor
            return torch.tensor([random.randrange(self.action_size)], device=device)
        else:
            # Compute the action using the policy net
            with torch.no_grad():
                state = torch.from_numpy(state).float().to(device)  # Convert state to a PyTorch tensor and move to the appropriate device
                state = state.unsqueeze(0)  # Add batch dimension if not already added
            q_values = self.policy_net(state)
            # Return the action as a single-element tensor
            return q_values.max(1)[1].view(1)  # Flatten to [1] instead of [1,1] or something similar

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    
        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()
    
        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        next_states = torch.from_numpy(next_states).cuda()
        dones = mini_batch[3]
        mask = torch.tensor(list(map(int, dones == False)), dtype=torch.uint8).cuda()

        # Your agent.py code here with double DQN modifications
        ### CODE ###"

        # Compute Q(s_t, a), the Q-value of the current state
        state_action_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    
        # Use the policy net to select the best action to take for the next states (argmax of Q values)
        next_actions = self.policy_net(next_states).max(1)[1]
    
        # Use the target net to calculate the Q values for the next states and selected actions by policy net
        next_state_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
    
        # Find maximum Q-value of action at next state from target net
        next_state_values = next_state_values.detach()  # detach to stop gradient backpropagation to target net
    
        # Compute expected Q values
        expected_state_action_values = rewards + (self.discount_factor * next_state_values * mask)
    
        # Compute the Huber Loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

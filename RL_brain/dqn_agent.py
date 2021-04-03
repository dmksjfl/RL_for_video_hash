import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
import os

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, hidden_state):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(hidden_state, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self, state_dim, update_interval = 10, q_lr=0.001, gamma = 0.7):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(state_dim), Net(state_dim)

        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=q_lr)
        self.loss_func = nn.MSELoss()
        self.update_interval = update_interval
        self.gamma = gamma

    def choose_action(self, state, action_dim):
        state = torch.FloatTensor(state.detach().cpu().numpy())
        action_value = self.eval_net.forward(state)
        action_value = torch.FloatTensor(action_value).squeeze(2)
        _, action = action_value.topk(action_dim, dim=1, largest=True, sorted=True)
        #print(action.shape)
        return action
    
    def random_act(self, batch_size = 100, max_frames = 30, outframe = 15):
        action = []
        for i in range(batch_size):
            frame_shuffle = np.arange(max_frames)
            np.random.shuffle(frame_shuffle)
            frame_shuffle = np.array(frame_shuffle)
            frame_shuffle = frame_shuffle[:outframe]
            action.append(frame_shuffle)
        return np.array(action)


    def learn(self, replay_buffer, batch_size):

        #update the parameters
        if self.learn_step_counter % self.update_interval ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        action = action.detach().cpu().unsqueeze(-1)
        state = state.detach().cpu(); next_state = next_state.detach().cpu(); reward = reward.detach().cpu()

        #q_eval
        q_eval = self.eval_net(state).gather(1, action).squeeze(-1)
        q_next = self.target_net(next_state).detach()
        q_target = reward + self.gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, file_name):
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        torch.save(self.eval_net.state_dict(), file_name + 'model_eval_net')
        torch.save(self.target_net.state_dict(), file_name + 'model_target_net')
        torch.save(self.optimizer, file_name + 'optimizer')
    
    def load_model(self, file_name):
        self.eval_net.load_state_dict(torch.load(file_name+'model_eval_net'))
        self.target_net.load_state_dict(torch.load(file_name+'model_target_net'))
        #self.optimizer.load_state_dict(torch.load(file_name+'optimizer'))

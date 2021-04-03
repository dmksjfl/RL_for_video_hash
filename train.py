import os
import torch
import argparse
import numpy as np
from utils.args import *
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from model import BERTLM
from optim_schedule import ScheduledOptim
import scipy.io as sio
import pickle
from data import get_train_loader, get_eval_loader
import h5py
from RL_brain.ReplayBuffer import ReplayBuffer
from RL_brain.dqn_agent import DQN

if not os.path.exists(file_path):
    os.makedirs(file_path)

best_optimizer_pth_path = file_path+'/best_optimizer.pth'
optimizer_pth_path = file_path+'/optimizer.pth'
print('Learning rate: %.4f' % lr)
infos = {}
infos_best = {}
histories = {}
if use_checkpoint is True and os.path.isfile(os.path.join(file_path, 'infos.pkl')):
    with open(os.path.join(file_path, 'infos.pkl')) as f:
        infos = pickle.load(f)

    if os.path.isfile(os.path.join(file_path, 'histories.pkl')):
        with open(os.path.join(file_path, 'histories.pkl')) as f:
            histories = pickle.load(f)

model = BERTLM(feature_size).cuda()
train_loader = get_train_loader(train_feat_path, sim_path, batch_size,shuffle = True)

itera = 0
epoch = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
replay_buffer = ReplayBuffer([max_frames,hidden_size], action_dim, device, replay_buffer_size)
agent = DQN(hidden_size)

if use_checkpoint:
    model.load_state_dict(torch.load(file_path + '/9288.pth'))
    itera = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
optimizer = Adam(model.parameters(), lr=lr)
optim_schedule = ScheduledOptim(optimizer, hidden_size, n_warmup_steps=10000)
if os.path.exists(best_optimizer_pth_path) and use_checkpoint:
    optimizer.load_state_dict(torch.load(optimizer_pth_path))

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

total_len = len(train_loader)

while True:
    epoch_reward = 0.
    decay_factor = 0.9  ** ((epoch)//lr_decay_rate)
    current_lr = max(lr * decay_factor,1e-4)
    set_lr(optimizer, current_lr) # set the decayed rate
    done = False
    for i, data in enumerate(train_loader, start=1):
        data = {key: value.cuda() for key, value in data.items()}
        _,_,hid1 = model.forward(data["visual_word"][:,:max_frames,:])
        _,_,hid2 = model.forward(data["visual_word"][:,max_frames:,:])

        if epoch < warm_up_epoch:
            action = agent.random_act(batch_size, max_frames, action_dim)
        else:
            action = agent.choose_action(hid1, action_dim)
            #print(action)
        #print(data['mask_input'].shape, data['visual_word'].shape, data['n1'].shape)
        
        state = hid1
        next_state = hid2
        newdata = {}
        tempdata = {}

        iter_batch = data["mask_input"].size(0)
        
        for key in ['mask_input', 'visual_word']:
            for z in range(iter_batch):
                tempdata[key] = data[key][z,action[z][0],:].unsqueeze(0)
                for k in range(action.shape[1]-1):
                    tempdata[key] = torch.cat((tempdata[key], data[key][z][action[z,k+1],:].unsqueeze(0)),0)
                    tempdata[key] = torch.cat((tempdata[key], data[key][z][action[z,k+1]+max_frames,:].unsqueeze(0)),0)
                if z ==0:
                    newdata[key] = tempdata[key].unsqueeze(0)
                else:
                    newdata[key] = torch.cat((newdata[key], tempdata[key].unsqueeze(0)),0)
            data[key] = newdata[key]
        
        optimizer.zero_grad()
        batchsize = data["mask_input"].size(0)
        bb1,frame1,hid1 = model.forward(data["mask_input"][:,:action_dim,:])
        bb2,frame2,hid2 = model.forward(data["mask_input"][:,action_dim:,:])
        bb1 = torch.mean(bb1,1)
        bb2 = torch.mean(bb2,1)
        sim = bb1.mul(bb2)
        sim = torch.sum(sim,1)/nbits
        #print(sim.shape, bb1.shape, hid1.shape)

        nei_loss = torch.sum((1*data["is_similar"].float()-sim)**2)/batchsize
        mask_loss = (torch.sum((frame1-data["visual_word"][:,:action_dim,:])**2)\
                +torch.sum((frame2-data["visual_word"][:,action_dim:,:])**2))\
                /(2*action_dim*feature_size*batchsize)
        
        mu_loss = (torch.sum((torch.mean(hid1,1)-data['n1'])**2)\
            +torch.sum((torch.mean(hid2,1)-data['n2'])**2))/(hidden_size*batchsize)
        loss = 0.92*mask_loss + 0.08*nei_loss+0.8*mu_loss

        loss.backward()
        optimizer.step()
        
        rewards = []
        for m in range(iter_batch):
            rewards.append(torch.exp(1-(1*data["is_similar"][m].float()-sim[m].detach())**2)-1)
        
        done_bool = float(done) if i == total_len-1 else 0
        for m in range(iter_batch):
            replay_buffer.add(state[m,:,:], action[m,:], next_state[m,:,:], rewards[m], done_bool)
        
        if epoch >= warm_up_epoch:
            agent.learn(replay_buffer, batch_size)
        
        epoch_reward += sum(rewards)
        
        # loss = mask_loss

        itera += 1
        infos['iter'] = itera
        infos['epoch'] = epoch
        if itera%10 == 0 or batchsize<batch_size:  
            print('Epoch:%d Step:[%d/%d] neiloss: %.2f maskloss: %.2f mu_loss: %.2f epoch_reward: %.2f' \
            % (epoch, i, total_len, nei_loss.data.cpu().numpy(),\
            mask_loss.data.cpu().numpy(),mu_loss.data.cpu().numpy(),epoch_reward.data.cpu().numpy()))


    torch.save(model.state_dict(), file_path + '/9288.pth')
    torch.save(optimizer.state_dict(), optimizer_pth_path)

    with open(os.path.join(file_path, 'infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)
    with open(os.path.join(file_path, 'histories.pkl'), 'wb') as f:
        pickle.dump(histories, f)
    epoch += 1
    #print(action.shape)
    if epoch>num_epochs:
        agent.save_model(agent_save_dir)
        break
    model.train()

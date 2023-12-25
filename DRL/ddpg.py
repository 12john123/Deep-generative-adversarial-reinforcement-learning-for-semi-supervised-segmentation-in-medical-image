import argparse
from itertools import count

import os, sys, random
import numpy as np
from DRL.model import (PolicyNetwork, ValueNetwork, SoftQNetwork)
from DRL.actor import Actor
from DRL.critic import Critic

#import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from paramter import *
from utils.util import *
from DRL.random_process import OrnsteinUhlenbeckProcess
criterion = nn.MSELoss()


class SAC(object):
    def __init__(self):

        #if args.seed > 0:
        self.seed(500)

        # Create Actor and Critic Network
        self.policy_net = PolicyNetwork()
        self.policy_net_optim = optim.Adam(self.policy_net.parameters(), lr=0.0001)

        self.t_policy_net=PolicyNetwork()
        hard_update(self.t_policy_net, self.policy_net)

        self.value_net = ValueNetwork()
        self.value_net_target = ValueNetwork()
        self.value_net_optim = optim.Adam(self.value_net.parameters(), lr=0.0001)

        self.soft_q_network = SoftQNetwork()
        self.soft_q_optimizer = optim.Adam(self.soft_q_network.parameters(), lr=0.0001)
        self.t_soft_q = SoftQNetwork()

        hard_update(self.value_net_target, self.value_net)
        hard_update(self.t_soft_q, self.soft_q_network)

        # Create replay buffer
        #self.memory = Memory()
        self.random_process = OrnsteinUhlenbeckProcess(size=4, theta=0.2, mu=0,
                                                       sigma=0.15)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()

        # Hyper-parameters
        self.batch_size = rl_batch_size
        self.tau = tau
        self.discount = 0.99
        #self.epsilon = 1.0
        self.depsilon = 1.0 / 50000
        self.nb_actions = 4
        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True

        #
        if USE_CUDA:
            self.cuda()
    
    def update_policy_unlabled(self,memory,k_num):
        '''
        用t 的q值v值 更新
        '''
        policy_losses = []

        for i in range(0, k_num*3):
            # Sample batch
            state,next_state, action, _, gan_score = memory.sample(1)

            lr=gan_score*0.001
            for param_group in self.value_net_optim.param_groups:
                param_group['lr'] = lr.item() 
            for param_group in self.policy_net_optim.param_groups:
                param_group['lr'] = lr.item()
            for param_group in self.soft_q_optimizer.param_groups:
                param_group['lr'] = lr.item()

            state_train = state
            next_state_train = next_state
            state_train = torch.FloatTensor(state_train).to(device)
            next_state_train = torch.FloatTensor(next_state_train).to(device)
            action_train = torch.FloatTensor(action).to(device)
            #reward_train = torch.FloatTensor(reward).to(device)

            expected_q_value = self.soft_q_network(state_train, action_train)
            expected_value = self.value_net(state_train)
            if multi_GPUs:
                new_action, log_prob, z, mean, log_std = self.policy_net.module.evaluate(state_train)
            else:
                new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state_train)
            #target_value = self.value_net_target(next_state_train)
            next_q_value = self.t_soft_q(state_train, action_train)     #reward_train + gamma * target_value

            q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

            expected_new_q_value = self.t_soft_q(state_train, new_action)
            #next_value = expected_new_q_value - log_prob
            next_value=self.value_net_target(next_state_train)
            value_loss = self.value_criterion(expected_value, next_value.detach())

            expected_value_t=self.value_net_target(state_train)
            log_prob_target = expected_new_q_value - expected_value_t
            policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

            mean_loss = mean_lambda * mean.pow(2).mean()
            std_loss = std_lambda * log_std.pow(2).mean()
            z_loss = z_lambda * z.pow(2).sum(1).mean()

            policy_loss += mean_loss + std_loss + z_loss
            policy_losses.append(to_numpy(policy_loss))
            self.soft_q_optimizer.zero_grad()
            q_value_loss.backward()
            q_value_loss=0
            self.soft_q_optimizer.step()

            self.value_net_optim.zero_grad()
            value_loss.backward()
            value_loss=0
            self.value_net_optim.step()

            self.policy_net_optim.zero_grad()
            policy_loss.backward()
            policy_loss=0
            self.policy_net_optim.step()

            for target_param, param in zip(self.value_net_target.parameters(),
                                           self.value_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau/10) + param.data * tau/10)
            
            for target_param, param in zip(self.t_policy_net.parameters(),
                                           self.policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau/10) + param.data * tau/10)
            
            for target_param, param in zip(self.t_soft_q.parameters(),
                                           self.soft_q_network.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau/10) + param.data * tau/10)

        policy_loss = np.mean(np.array(policy_losses))

        return policy_loss/(k_num*3)
        

        
    def update_policy(self, lr,memory,inverse_ReplayMem,inverse_rate):

        for param_group in self.value_net_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.policy_net_optim.param_groups:
            param_group['lr'] = lr
        for param_group in self.soft_q_optimizer.param_groups:
            param_group['lr'] = lr

        policy_losses = []

        for i in range(0, training_num_per_iteration):
            # Sample batch
            state,next_state, action, reward, terminal = memory.sample(int(rl_batch_size*(1-inverse_rate))+1)

            state_train = state
            next_state_train = next_state
            state_train = torch.FloatTensor(state_train).to(device)
            next_state_train = torch.FloatTensor(next_state_train).to(device)
            action_train = torch.FloatTensor(action).to(device)
            reward_train = torch.FloatTensor(reward).to(device)

            state,next_state, action, reward, terminal = inverse_ReplayMem.sample(int(rl_batch_size*(inverse_rate)))

            state_train1 = state
            next_state_train1 = next_state
            state_train1 = torch.FloatTensor(state_train1).to(device)
            next_state_train1 = torch.FloatTensor(next_state_train1).to(device)
            action_train1 = torch.FloatTensor(action).to(device)
            reward_train1 = torch.FloatTensor(reward).to(device)

            state_train=torch.cat([state_train,state_train1],dim=0)
            next_state_train=torch.cat([next_state_train,next_state_train1],dim=0)
            action_train=torch.cat([action_train,action_train1],dim=0)
            reward_train=torch.cat([reward_train,reward_train1],dim=0)


            expected_q_value = self.soft_q_network(state_train, action_train)
            expected_value = self.value_net(state_train)
            if multi_GPUs:
                new_action, log_prob, z, mean, log_std = self.policy_net.module.evaluate(state_train)
            else:
                new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state_train)
            target_value = self.value_net_target(next_state_train)
            next_q_value = reward_train + gamma * target_value

            q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

            expected_new_q_value = self.soft_q_network(state_train, new_action)
            next_value = expected_new_q_value - log_prob
            value_loss = self.value_criterion(expected_value, next_value.detach())

            log_prob_target = expected_new_q_value - expected_value
            policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

            mean_loss = mean_lambda * mean.pow(2).mean()
            std_loss = std_lambda * log_std.pow(2).mean()
            z_loss = z_lambda * z.pow(2).sum(1).mean()

            policy_loss += mean_loss + std_loss + z_loss
            policy_losses.append(to_numpy(policy_loss))
            self.soft_q_optimizer.zero_grad()
            q_value_loss.backward()
            self.soft_q_optimizer.step()

            self.value_net_optim.zero_grad()
            value_loss.backward()
            self.value_net_optim.step()

            self.policy_net_optim.zero_grad()
            policy_loss.backward()
            self.policy_net_optim.step()

            for target_param, param in zip(self.value_net_target.parameters(),
                                           self.value_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
            for target_param, param in zip(self.t_policy_net.parameters(),
                                           self.policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
            for target_param, param in zip(self.t_soft_q.parameters(),
                                           self.soft_q_network.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        policy_loss = np.mean(np.array(policy_losses))

        return policy_loss/rl_batch_size

    def eval(self):
        self.policy_net.eval()
        self.value_net_target.eval()
        self.value_net.eval()
        self.soft_q_network.eval()

    def cuda(self):
        self.policy_net.to(device)     #.cuda()
        self.value_net_target.to(device)
        self.value_net.to(device)
        self.soft_q_network.to(device)
        self.t_policy_net.to(device)
        self.t_soft_q.to(device)


    def observe(self, r_t, s_t1):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, s_t1)
            self.s_t = s_t1

    def random_action(self,size):
        actions = []
        for i in range(0, size):
            action = np.random.uniform(-1., 1., self.nb_actions)
            actions.append(action)
        actions = np.array(actions)
        self.a_t = actions
        return actions

    def select_action(self, s_t, decay_epsilon=True):

        action = self.t_policy_net.get_action(s_t)

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()
        

    def save(self,i,directory):
        torch.save(self.policy_net.state_dict(), directory + str(i)+'policy_net_liver.pth')
        torch.save(self.value_net.state_dict(), directory + str(i)+'value_net_liver.pth')
        torch.save(self.soft_q_network.state_dict(), directory + str(i)+'soft_q_network_liver.pth')
        #torch.save(self.critic_target.state_dict(), directory + str(i)+'critic.pth')
    

        
    def load(self,i,directory):
        self.policy_net.load_state_dict(torch.load(directory + str(i)+'policy_net_liver.pth',map_location=device))
        self.value_net.load_state_dict(torch.load(directory + str(i)+'value_net_liver.pth',map_location=device))
        self.soft_q_network.load_state_dict(torch.load(directory + str(i)+'soft_q_network_liver.pth',map_location=device))
        self.value_net_target.load_state_dict(torch.load(directory + str(i)+'value_net_liver.pth',map_location=device))
        hard_update(self.t_soft_q, self.soft_q_network)
        hard_update(self.t_policy_net, self.policy_net)
        print("====================================")
        print("model has been loaded...")
        print("====================================")
    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)    


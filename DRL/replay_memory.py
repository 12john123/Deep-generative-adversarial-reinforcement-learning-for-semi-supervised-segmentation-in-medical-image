import random
import numpy as np
import torch
from paramter import *
class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state,next_state, action, reward,done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state,next_state, action, reward,done)
        self.position = (self.position + 1) % self.capacity
    
    def muti_push(self, state,next_state, action, reward,done):
        for i in range(state.shape[0]):
            self.push(state[i],next_state[i],action[i],reward[i],done[i])
    
    def muti_push_previous(self, state,next_state, action, reward,done):
        for i in range(next_state.shape[0]):
            for j in range(5):
                self.push(state[i*5+j],next_state[i],action[i*5+j],reward[i*5+j],done[i*5+j])
    #state next_state action reward done

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state = map(np.stack, zip(*batch))
        order = 0
        state_old_train = np.zeros([batch_size, 2, wid_img_def, hei_img_def],np.float32)
        state_target_train = np.zeros([batch_size, 2, wid_img_def, hei_img_def],np.float32)
        action_train = np.zeros([batch_size, 4],np.float32)
        reward_train = np.zeros([batch_size, 1],np.float32)
        done_train=np.zeros([batch_size, 1],np.float32)
        for memory in batch:
            state_old,state_new, action, reward,done  = memory
            state_old_train[order] = state_old
            state_target_train[order] = state_new
            action_train[order] = action
            # print(action_train.size())
            reward_train[order] = reward
            done_train[order]=done
            order += 1
        return state_old_train,state_target_train, action_train, reward_train,done_train

    def sample_for_combine(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # state, action, reward, next_state = map(np.stack, zip(*batch))
        order = 0
        state_old_train = torch.FloatTensor(batch_size, 4, wid_img_def, hei_img_def)
        state_target_train = torch.FloatTensor(batch_size, 4, wid_img_def, hei_img_def)
        action_train = torch.FloatTensor(batch_size, 5)
        reward_train = torch.FloatTensor(batch_size, 1)
        done_train=torch.FloatTensor(batch_size, 1)
        for memory in batch:
            state_old,state_new, action, reward,done  = memory
            state_old_train[order] = state_old
            state_target_train[order] = state_new
            action_train[order] = torch.FloatTensor(action)
            # print(action_train.size())
            reward_train[order] = reward
            done_train[order]=done
            order += 1
        return state_old_train,state_target_train, action_train, reward_train,done_train 

    def __len__(self):
        return len(self.buffer)

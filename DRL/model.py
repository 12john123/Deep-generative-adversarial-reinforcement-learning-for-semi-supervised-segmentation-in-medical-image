import numpy as np
from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
from paramter import *
import DRL.ResNet as ResNet

LOG_SIG_MAX = 1
LOG_SIG_MIN = -1
epsilon = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)


class Actor(nn.Module):
    def __init__(self, ):
        super(Actor, self).__init__()
        self.resNet = ResNet.ResNet18(2)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop2d = nn.Dropout2d(0.2)
        self.drop = nn.Dropout(0.1)

        self.linear1 = nn.Linear(4096, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 64)
        self.linear4 = nn.Linear(64, 3)
        # self.linear5 = nn.Linear(128, 64)
        # self.linear6 = nn.Linear(64, 16)
        # self.linear7 = nn.Linear(16, 3)
        self.apply(weights_init_)

    def forward(self, x):
        x = self.resNet(x)
        bs, ch, hei, wid = x.size()
        x = x.view(bs, ch * hei * wid)
        x = self.drop(self.relu(self.linear1(x)))
        x = self.drop(self.relu(self.linear2(x)))
        x = self.drop(self.relu(self.linear3(x)))
        x = self.linear4(x)
        x = torch.sigmoid(x)
        return x


class Critic(nn.Module):
    def __init__(self,):
        super(Critic, self).__init__()
        self.resNet = ResNet.ResNet18(2)
        self.prelu = nn.Tanh()
        self.relu = nn.Tanh()
        self.drop2d = nn.Dropout2d(0.2)
        self.drop = nn.Dropout(0.1)

        self.linear1 = nn.Linear(16387, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 64)
        self.linear4 = nn.Linear(64, 1)

        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 16)
        self.linear7 = nn.Linear(16, 1)

        self.apply(weights_init_)

    def forward(self, xs):
        x, a = xs
        x = self.resNet(x)
        bs, ch, hei, wid = x.size()
        x = x.view(bs, ch * hei * wid)
        x = torch.cat((x, a), 1)
        x = self.drop(self.relu(self.linear1(x)))
        x = self.drop(self.relu(self.linear2(x)))
        x = self.drop(self.relu(self.linear3(x)))
        x = self.linear4(x)
        return x


########### soft actor-critic #################

class ValueNetwork(nn.Module):
    def __init__(self,):
        super(ValueNetwork, self).__init__()
        self.resNet = ResNet.ResNet18(2)
        self.prelu = nn.Tanh()
        self.relu = nn.Tanh()
        self.drop2d = nn.Dropout2d(0.2)
        self.drop = nn.Dropout(0.2)
        self.linear5 = nn.Linear(4096, 1024)
        self.linear6 = nn.Linear(1024, 512)
        self.linear7 = nn.Linear(512, 256)
        self.linear8 = nn.Linear(256, 4)

    def forward(self, x):

        x = self.resNet(x)
        bs, ch, hei, wid = x.size()
        x = x.view(bs, ch * hei * wid)
        x = self.drop(self.relu(self.linear5(x)))
        x = self.drop(self.relu(self.linear6(x)))
        x = self.relu(self.linear7(x))
        x = self.linear8(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self,):
        super(SoftQNetwork, self).__init__()
        self.resNet = ResNet.ResNet18(2)
        self.prelu = nn.Tanh()
        self.relu = nn.Tanh()
        self.drop2d = nn.Dropout2d(0.2)
        self.drop = nn.Dropout(0.2)

        self.linear1 = nn.Linear(4096+4, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 4)

        self.apply(weights_init_)

    def forward(self, x, action):
        x = self.resNet(x)
        bs, ch, hei, wid = x.size()
        x = x.view(bs, ch * hei * wid)
        x = torch.cat((x, action), 1)
        x = self.drop(self.relu(self.linear1(x)))
        x = self.drop(self.relu(self.linear2(x)))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self,):
        super(PolicyNetwork, self).__init__()
        self.resNet = ResNet.ResNet18(2)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.drop2d = nn.Dropout2d(0.2)
        self.drop = nn.Dropout(0.2)

        self.linear1 = nn.Linear(4096, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)

        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 16)
        self.linear7 = nn.Linear(16, 4)

        self.linear8 = nn.Linear(128, 64)
        self.linear9 = nn.Linear(64, 16)
        self.linear10 = nn.Linear(16, 4)
        self.tanh = nn.Tanh()
        self.noise = torch.Tensor(4)
        self.apply(weights_init_)

    def forward(self, x):

        x = self.resNet(x)
        bs, ch, hei, wid = x.size()
        x = x.view(bs, ch * hei * wid)
        x = self.drop(self.relu(self.linear1(x)))
        x = self.drop(self.relu(self.linear2(x)))
        x = self.drop(self.relu(self.linear3(x)))
        x = self.drop(self.relu(self.linear4(x)))
        mean = self.linear5(x)
        mean = self.linear6(mean)
        mean = self.linear7(mean)
        # mean = self.tanh(mean)
        log_std = self.linear8(x)
        log_std = self.linear9(log_std)
        log_std = self.linear10(log_std)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        #state = torch.FloatTensor(state).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.detach().cpu().numpy()
        return action


class Discriminator(nn.Module):
    def __init__(self, input_channel, num_classes, ndf=64):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(ndf),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(ndf * 2),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(ndf * 4),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(ndf * 8),
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(ndf * 8),
            nn.Conv2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(ndf * 4),
            nn.Conv2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            )
        self.fc1 = nn.Linear(512, 256)
        self.fc1_norm = nn.InstanceNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc1_norm = nn.InstanceNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc1_norm = nn.InstanceNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.fc1_norm = nn.InstanceNorm1d(32)
        self.fc5 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        b, c, w, h = x.shape
        x = x.view(b, c * w * h)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x

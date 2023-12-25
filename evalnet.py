
import torch
import torch.nn as nn
from ResNet import basic_ResNet50
from paramter import wid_img_def



class evalNet(nn.Module):
    def __init__(self, n_channels):
        super(evalNet, self).__init__()
        self.n_channels = n_channels
        
        self.seq=nn.Sequential(
            nn.Conv2d(self.n_channels,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(4),

            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(4),

            nn.Conv2d(256,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )

        #self.resnet18=basic_ResNet50(input_channel=n_channels)
        self.fc1=nn.Linear(2048,1024)
        self.fc2=nn.Linear(1024,256)
        self.fc3=nn.Linear(256,64)
        self.fc4=nn.Linear(64,1)

    def forward(self,x):
        #x=self.resnet18(x)
        x=self.seq(x)
        x=x.flatten(start_dim=1)

        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        x=torch.sigmoid(x)

        return x

""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch,torchvision
import torch.nn as nn
from collections import OrderedDict
import sys
sys.path.append("..")
from paramter import *
import ResNet
from torchvision.models.detection.roi_heads import paste_masks_in_image

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class FCN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=64, dropout_p= 0.5):
        super().__init__()
        features = init_features

        #self.fpn=torchvision.models.
        self.resnet=ResNet.basic_ResNet50(in_channels)        #torchvision.models.resnet50(pretrained=True)

        self.roialign=torchvision.ops.RoIAlign( output_size=14, spatial_scale=1.0, sampling_ratio=-1)

        self.roialign_gt=torchvision.ops.RoIAlign( output_size=28, spatial_scale=1.0, sampling_ratio=-1)

        self.out_conv=nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64,16,3,1,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16,4,3,1,1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),

            nn.Conv2d(4,1,3,1,1),
            nn.Sigmoid()
        )

        self.weight_init()   
    
    def forward(self,img,box):
        feature=self.resnet(img)
        box_t=torch.FloatTensor(box).to(device)
        box_t=torch.split(box_t,1,dim=0)
        roi_feature=self.roialign(feature,box_t)

        roi_seg=self.out_conv(roi_feature)

        return roi_seg
    
    def get_roi_gt(self,gt,box):
        box_t=torch.FloatTensor(box).to(device)
        box_t=torch.split(box_t,1,dim=0)
        
        roi_gt=self.roialign_gt(gt,box_t)

        return roi_gt
    
    def get_mask(self,roi_mask,box):
        box_t=torch.FloatTensor(box).to(device)
        #box_t=torch.split(box_t,1,dim=0)
        mask=paste_masks_in_image(roi_mask,boxes=box_t,img_shape=(256,256),padding=1)

        return mask
    
    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    mean = 0
                    # standard deviation based on a 3*3 convolution
                    std =  (2/(3*3* m.out_channels))**(0.5)
                    self.normal_init(m, self.mean, self.std)
            except:
                pass



class UNet_2D(nn.Module):
    #2D UNet architecture
    def __init__(self, in_channels=1, out_channels=1, init_features=64, dropout_p= 0.5):
        super().__init__()
        features = init_features
        
        # Encoding layers
        self.encoder1 = UNet_2D._block(in_channels, features)   
        self.encoder2 = UNet_2D._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_2D._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_2D._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck layer
        self.bottleneck = UNet_2D._block(features * 8, features * 16)

        # Decoding layers
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet_2D._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet_2D._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet_2D._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet_2D._block(features * 2, features)

        # output layer
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

        # Max Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(dropout_p)

        self.weight_init()   

        self.resnet=ResNet.basic_ResNet50(1)        #torchvision.models.resnet50(pretrained=True)

        self.roialign=torchvision.ops.RoIAlign( output_size=14, spatial_scale=1.0, sampling_ratio=-1)


    
    def get_state(self,img,box):
        feature=self.resnet(img)
        box_t=torch.FloatTensor(box).to(device)
        box_t=torch.split(box_t,1,dim=0)
        roi_feature=self.roialign(feature,box_t)

        return roi_feature


    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.weight.data.normal_(mean, std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    # Weight initialization 
    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    mean = 0
                    # standard deviation based on a 3*3 convolution
                    std =  (2/(3*3* m.out_channels))**(0.5)
                    self.normal_init(m, self.mean, self.std)
            except:
                pass
            
    # Weight standardization: A normalization to be used with group normalization (micro_batch)
    def WS(self):
        for block in self._modules:
            if isinstance(block, nn.MaxPool2d) or isinstance(block, nn.ConvTranspose2d):
                pass
            else:
                for m in block:
                    if isinstance(m, nn.Conv2d):
                        weight = m.weight
                        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                      keepdim=True).mean(dim=3, keepdim=True)
                        weight = weight - weight_mean
                        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
                        weight = weight / std.expand_as(weight)
                        m.weight.data = weight
          

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        p1 = self.dropout(self.pool(enc1))
        enc2 = self.encoder2(p1)
        p2 = self.dropout(self.pool(enc2))
        enc3 = self.encoder3(p2)
        p3 = self.dropout(self.pool(enc3))
        enc4 = self.encoder4(p3)
        p4 = self.dropout(self.pool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(p4)      

        # Decoding path
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout(self.upconv4(bottleneck))
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.dropout(self.upconv3(dec4))
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.dropout(self.upconv2(dec3))
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.dropout(self.upconv1(dec2))
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        #self.WS()
        # Output
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_features, out_features):               
        return nn.Sequential(OrderedDict([
                    ("conv1",nn.Conv2d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=3,
                        padding=1,
                        bias=False)),
                    ("norm1", nn.BatchNorm2d(num_features=out_features)),
                    #("relu1", nn.ReLU(inplace=True)),
                    ("swish1", nn.SiLU(inplace=True)),
                    ("conv2",nn.Conv2d(
                        in_channels=out_features,
                        out_channels=out_features,
                        kernel_size=3,
                        padding=1,
                        bias=False)),
                    ("norm2", nn.BatchNorm2d(num_features=out_features)),
                    #("relu2", nn.ReLU(inplace=True))
                    ("swish2", nn.SiLU(inplace=True))
                ]))


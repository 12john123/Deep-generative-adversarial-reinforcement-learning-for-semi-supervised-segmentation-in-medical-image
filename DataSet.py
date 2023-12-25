import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2
import random

trans=transforms.ToTensor()

transforms=transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.5],std=[.5])
    transforms.Resize([256,256])
])


class annotatedSet(Dataset):
    def __init__(self,DataFilePath,is_augment) -> None:
        super().__init__()
        DataImgs=os.listdir(DataFilePath)
        self.is_augment=is_augment
        self.DataImgs=[os.path.join(DataFilePath,k) for k in DataImgs]

    def __len__(self):
        return len(self.DataImgs)//2

    def augment(self,image,flipCode):
        flip=cv2.flip(image,flipCode)
        return flip

    def __getitem__(self, index):
        index*=2
        if self.DataImgs[index].find('mask')!=-1:
            index-=1
        DataImgPath=self.DataImgs[index]
        lpath=DataImgPath.replace('.png','_mask.png')
        LabelImgPath=lpath

        pil_Dimg=Image.open(DataImgPath)
        
        pil_Limg=Image.open(LabelImgPath)
        # if pil_Dimg.mode!='L':
        #         pil_Dimg=pil_Dimg.convert('L')
        #if pil_Limg.mode!='L':
        #      pil_Limg=pil_Limg.convert('L')
        data=transforms(pil_Dimg)
        
        #data = torch.autograd.Variable(torch.unsqueeze(data, dim=0).float(), requires_grad=False)
        #tf=transforms.ToTensor()
        label=transforms(pil_Limg)
        #label = torch.autograd.Variable(torch.unsqueeze(label, dim=0).float(), requires_grad=False)
        #label=F.relu(label)
        #data=F.relu(data)
        #数据增强
        if(self.is_augment):
            flipCode = random.choice([-1, 0, 1, 2])
            data=data.numpy()
            label=label.numpy()
            if flipCode != 2:
                data = self.augment(data, flipCode)
                label = self.augment(label, flipCode)

        return data,label

class notannotatedSet(Dataset):
    def __init__(self,DataFilePath,is_augment) -> None:
        super().__init__()
        DataImgs=os.listdir(DataFilePath)
        self.is_augment=is_augment
        self.DataImgs=[os.path.join(DataFilePath,k) for k in DataImgs]

    def __len__(self):
        return len(self.DataImgs)

    def augment(self,image,flipCode):
        flip=cv2.flip(image,flipCode)
        return flip

    def __getitem__(self, index):
        DataImgPath=self.DataImgs[index]
        
        pil_Dimg=Image.open(DataImgPath)
        
        data=transforms(pil_Dimg)
        data=F.relu(data)
        #数据增强
        if(self.is_augment):
            flipCode = random.choice([-1, 0, 1, 2])
            data=data.numpy()
            if flipCode != 2:
                data = self.augment(data, flipCode)

        return data

class mixdSet(Dataset):
    def __init__(self,anFilePath,labelFilePath,notanFilePath,is_augment) -> None:
        super().__init__()
        anImgs=os.listdir(anFilePath)
        notanImgs=os.listdir(notanFilePath)
        lableImgs=os.listdir(labelFilePath)

        self.is_augment=is_augment
        self.anlist=[os.path.join(anFilePath,k) for k in anImgs]
        self.notanlist=[os.path.join(notanFilePath,k) for k in notanImgs]
        self.labellist=[os.path.join(labelFilePath,k) for k in lableImgs]

        self.anlen=len(self.anlist)
        self.notanlen=len(self.notanlist)

    def __len__(self):
        return  500        #self.anlen

    def augment(self,image,flipCode):
        flip=cv2.flip(image,flipCode)
        return flip

    def __getitem__(self, index):
        
        DataImgPath=self.anlist[index]
        
        #LabelImgPath=self.labellist[index]
        #LabelImgPath=DataImgPath.replace('.png','_mask.png').replace('annotated','label')
        notan_idx=random.randint(0,self.notanlen-1)
        notanPath=self.notanlist[notan_idx]
        
        #
        #print(LabelImgPath)

        pil_Dimg=cv2.imread(DataImgPath)[:,:,-1] #.transpose(1,0)
        #print(DataImgPath)
        pil_Limg=cv2.imread(DataImgPath.replace('CT_','M_').replace('annotated','label'))[:,:,-1]*255
        pil_Limg=cv2.flip(pil_Limg,1)
        #print(np.max(pil_Dimg))
        #print(np.min(pil_Dimg))
        

        notan_img=cv2.imread(notanPath)[:,:,-1]

        data=transforms(pil_Dimg)

        label=transforms(pil_Limg)

        notan_img=transforms(notan_img)

        #数据增强
        if(self.is_augment):
            flipCode = random.choice([-1, 0, 1, 2])
            data=data.numpy()
            label=label.numpy()
            notan_img=notan_img.numpy()
            if flipCode != 2:
                data = self.augment(data, flipCode)
                label = self.augment(label, flipCode)
                notan_img=self.augment(notan_img,flipCode)
        return data,label,notan_img



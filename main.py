import os
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import torch,torchvision
import random
import numpy as np
from DRL.replay_memory import ReplayBuffer
from utils import DataSet
from torch.utils.data import DataLoader,random_split
from utils.util import *

from DRL.ddpg import SAC
from torch import inverse, optim
import cv2
import sys, os
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from unet.unet_model import UNet,UNet_2D,FCN
import evalnet
import dice
from torch.autograd import Variable
from loss import TverskyLoss
from dice import DiceLoss
import copy
from ResUNet import ResUNet
from hausdorff import hausdorff_distance
from sklearn.metrics import cohen_kappa_score,confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
torch.cuda.manual_seed(0)
e_start_epoch=0
flag1=0
gan_rate=1.0
inverse_rate=0.5

train_rate=0.3
label_rate=0.5

directory=str(train_rate).replace('.','')+'_'+str(label_rate).replace('.','')+'/'

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        
def adjust_gamma(image, gamma=1.0):
    invgamma = 1/gamma
    brighter_image = np.array(np.power((image/255), invgamma)*255, dtype=np.uint8)
    return brighter_image

def mixed_training(sac,generator,discriminator,ReplayMem,label_loader,unlabel_loader,val_loader,epoch,target_d,target_g,dis_d,dis_d_target,inverse_ReplayMem): 

    lr = 3e-4

    
    pob=0
    
    if epoch>10:
      dlr=0.00001
      d_seg_lr=0.00001
    else:
      dlr=0.00001
      d_seg_lr=0.00001
    
    
    Tversky=TverskyLoss().to(device)  
    
    bcecriterion=torch.nn.BCEWithLogitsLoss().to(device)
    bce=torch.nn.BCEWithLogitsLoss().to(device)
    
    generator_optim=optim.Adam(generator.parameters(),lr=0.0001)
    discriminator_optim=optim.RMSprop(discriminator.parameters(),lr=dlr)
    dis_d_optim=optim.RMSprop(dis_d.parameters(),lr=d_seg_lr)
    
    epoch_segloss=0.0
    epoch_actor_loss=0.0
    epoch_critic_loss=0.0
    epoch_unlable_critic_loss=0.0
    epoch_reward=0.0
    epoch_seg_gan_reward=0.0
    epoch_det_gan_reward=0.0
    epoch_boxreward=0.0
    epoch_iou=0.0
    epoch_ap=0.0
    not_update=0

    i_count=0

    img_num=0

    for idx,((an,label),(un_an,un_label)) in enumerate(zip(label_loader,unlabel_loader)):
        #break
        
        
        an=an.to(device)
        label=label.to(device)
        notan=un_an.to(device)
        gt_box=mutibox_from_mask(label)
        for i in gt_box:
            i[0]=np.max([0,i[0]-10])
            i[1]=np.max([0,i[1]-10])
            i[2]=np.min([127,i[2]+10])
            i[3]=np.min([127,i[3]+10])

        an_batch_size=an.size(0)
        notan_batch_size=notan.size(0)

        img_num+=an_batch_size
        img_num+=notan_batch_size

        conbine=torch.cat([an,notan],dim=0)    

        boxls=[]
        for j in range(an_batch_size+notan_batch_size):  #
          boxls.append([55.,55.,70.,70.])
        box=boxls

        global flag1
        flag1+=1
        
        done=0
        state=get_state(conbine,box)
        

          
        count=0
        while done==0:
            count+=1
            if count==30:
                done=1
            
            if flag1<10:
              action=sac.random_action(state.shape[0])
            else:
              action=sac.select_action(state)
            next_box=execute_mutiaction(box,action) 

            if idx==1:
               print(count,next_box[0])
            
            next_state=get_state(conbine,next_box)
            
            box_reward,iou_mean1,ap_mean,apls,iouls=get_boxreward(gt_box,box,next_box)
            



            with torch.no_grad():
              seg_state=get_state_seg(conbine,next_box)  
              seg=target_g(seg_state)
              seg_new=dis_d_target(torch.cat([conbine,seg],dim=1))
            
            seg_reward=get_ganreward(seg_new)
            
            with torch.no_grad():
                gan_scorels=target_d(next_state)
            gan_reward=get_ganreward(gan_scorels)

            if count==30:
              epoch_iou+=iou_mean1
              epoch_ap+=ap_mean
              i_count+=1

            global gan_rate
            if gan_rate>0.35:
              gan_rate-=0.00001
            reward=gan_reward +seg_reward     #box_reward.unsqueeze_(dim=1).to(device)
            epoch_reward+=reward.float().mean().detach().cpu().numpy()
            epoch_seg_gan_reward+=seg_reward.float().mean().detach().cpu().numpy()
            epoch_det_gan_reward+=gan_reward.float().mean().detach().cpu().numpy()

            epoch_boxreward+=box_reward[0:an_batch_size].float().mean().detach().cpu().numpy()

            
            if done==1:
                over=np.ones([state.shape[0]])
            else :
                over=np.zeros([state.shape[0]])

            ReplayMem.muti_push(state.clone().detach().cpu().numpy(),next_state.clone().detach().cpu().numpy(),action,reward.clone().detach().cpu().numpy(),over)
            state=next_state
            box=next_box

            
            #update g
            if count>25:
              seg_state=get_state_seg(conbine,next_box)  #next_state*next_state[:,1,:,:]
              seg=generator(seg_state)
              
                              
              seg_new=dis_d(torch.cat([conbine,seg],dim=1))
              label_real = Variable(torch.Tensor(an_batch_size+notan_batch_size).fill_(1).to(device), requires_grad=False)#
              
              refineseg_loss=Tversky(seg[0:an_batch_size],label)
              bceloss=bcecriterion(seg[0:an_batch_size],label)
              generator_loss=bce(seg_new.squeeze(dim=1),label_real) 
              
              
              
              epoch_segloss+=(refineseg_loss.item()+bceloss.item())

              generator_optim.zero_grad()
              generator_loss+=(refineseg_loss+bceloss)
              generator_loss.backward()
              generator_optim.step()                  
              soft_update(target_g,generator,0.001)
            #real
            if count<=20:
              real=torch.cat([an,label],dim=1)
              real_score=dis_d(real)
              label_real = Variable(torch.Tensor(an_batch_size).fill_(1).to(device), requires_grad=False)
              

              #fake
              refine_conbine=torch.cat([conbine,seg],dim=1)  ##
              fake=refine_conbine.detach()[an_batch_size:]
              fake_score=dis_d(fake)

              label_fake=Variable(torch.Tensor(notan_batch_size).fill_(0).to(device), requires_grad=False) # an_batch_size+
              real_loss=bce(real_score.squeeze(dim=1),label_real)
              fake_loss=bce(fake_score.squeeze(dim=1),label_fake)
              s_gp=min(an_batch_size,notan_batch_size)
              gp=cal_gradient_penalty(dis_d,real[0:s_gp],fake[0:s_gp],s_gp)
              

              dis_d_optim.zero_grad()
              d_loss=real_loss+fake_loss  +gp
              
              d_loss.backward()
              dis_d_optim.step()
              soft_update(dis_d_target,dis_d,0.001)

              
             #update d
             
              real=get_state(an,gt_box)
              real_score=discriminator(real)
              label_real = Variable(torch.Tensor(an_batch_size).fill_(1).to(device), requires_grad=False)
              
  
              #fake
              fake=get_state(conbine,box)  
              fake_score=discriminator(fake)
        
              label_fake_un=Variable(torch.Tensor(notan_batch_size).fill_(0).to(device), requires_grad=False)
              fake_loss_notan=bce(fake_score[an_batch_size:].squeeze(dim=1),label_fake_un)
              
              real_loss=bce(real_score.squeeze(dim=1),label_real)
              gp=cal_gradient_penalty(discriminator,real[0:s_gp],fake[0:s_gp],s_gp)
              
  
              discriminator_optim.zero_grad()
              d_loss=real_loss+fake_loss_notan  +gp
  
              d_loss.backward()
              discriminator_optim.step()
              soft_update(target_d,discriminator,0.001)

            

        # inverse rl
        an=an.detach()
        label=label.detach()
        notan=notan.detach()
        gt_box=mutibox_from_mask(label)
        for i in gt_box:
            i[0]=np.max([0,i[0]-10])
            i[1]=np.max([0,i[1]-10])
            i[2]=np.min([127,i[2]+10])
            i[3]=np.min([127,i[3]+10])

        an_batch_size=an.size(0)
        notan_batch_size=notan.size(0)

        conbine=an  #torch.cat([an,notan],dim=0)    


        box=copy.deepcopy(gt_box)
        state=get_state(conbine,box)
        done=0
        count=0
        while done==0:
            count+=1
            if count==30:
                done=1
            
            action=sac.random_action(state.shape[0]*5)
            previous_box=execute_mutiaction_inverse(box,action) #,over
            previous_state=get_state_previous(conbine,previous_box)


            with torch.no_grad():
                gan_scorels=target_d(state)

                seg_state=get_state_seg(conbine,box)  #next_state*next_state[:,1,:,:]
                seg=target_g(seg_state)

                seg_new=dis_d_target(torch.cat([conbine,seg],dim=1))
            gan_reward=get_ganreward_previous(gan_scorels)
            seg_reward=get_ganreward_previous(seg_new)

            if count <5:
              seg_state=get_state_seg(conbine,box)  #next_state*next_state[:,1,:,:]
              seg=generator(seg_state)

                  
              seg_new=dis_d(torch.cat([conbine,seg],dim=1))
              label_real = Variable(torch.Tensor(an_batch_size).fill_(1).to(device), requires_grad=False)
              
              refineseg_loss=Tversky(seg[0:an_batch_size],label)
              generator_loss=bce(seg_new.squeeze(dim=1),label_real)
              bceloss=bcecriterion(seg[0:an_batch_size],label)
              
              epoch_segloss+=refineseg_loss.item()

              generator_optim.zero_grad()
              generator_loss+=(refineseg_loss+bceloss)#+consistency_loss

              generator_loss.backward()
              generator_optim.step()                  
              soft_update(target_g,generator,0.001)

              
              # update discriminator

              #real
              real=torch.cat([an,label],dim=1)
              real_score=dis_d(real)
              label_real = Variable(torch.Tensor(an_batch_size).fill_(1).to(device), requires_grad=False)
              
              #fake
              refine_conbine=torch.cat([conbine,seg],dim=1)  ##
              fake=refine_conbine.detach()
              fake_score=dis_d(fake) 
              real_loss=bce(real_score.squeeze(dim=1),label_real)
              gp=cal_gradient_penalty(dis_d,real,fake,an_batch_size)
              
              if flag1>0:
                dis_d_optim.zero_grad()
                d_loss=real_loss +gp
                
                if random.uniform(0,1) >pob :
                  d_loss=d_loss
                  d_loss.backward()
                  dis_d_optim.step()
                  soft_update(dis_d_target,dis_d,0.001)
                else:
                  d_loss=0
                  real_loss=0
                  fake_loss=0
                  gp=0

              real=get_state(an,gt_box)
              real_score=discriminator(real)

              label_real = Variable(torch.Tensor(an_batch_size).fill_(1).to(device), requires_grad=False)


              real_loss=bce(real_score.squeeze(dim=1),label_real)

              gp=cal_gradient_penalty(discriminator,real,fake,an_batch_size)
                    
              if flag1>0:
                discriminator_optim.zero_grad()
                d_loss=real_loss+gp  
                
                  
                if random.uniform(0,1) >pob :
                    d_loss=d_loss
                    d_loss.backward()
                    discriminator_optim.step()
                    soft_update(target_d,discriminator,0.001)
                else:
                    d_loss=0
                    real_loss=0
                    fake_loss=0
                    gp=0


            reward=gan_reward+seg_reward    

            if count==1:
                over=np.ones([previous_state.shape[0]])
            else :
                over=np.zeros([previous_state.shape[0]])

            inverse_ReplayMem.muti_push_previous(previous_state.clone().detach().cpu().numpy()[0:an_batch_size*5],state.clone().detach().cpu().numpy()[0:an_batch_size],action[0:an_batch_size*5],reward.clone().detach().cpu().numpy()[0:an_batch_size*5],over[0:an_batch_size*5])
            
            statels=[]
            new_box=[]
            for i in range(state.size(0)):
                iou_max=0
                iou=0
                for j in range(5):
                    if gan_reward[i*5+j]>0.2:
                        iou=gan_reward[i*5+j]
                        iou_max= i*5+j
                statels.append(previous_state[iou_max].clone().detach())
                new_box.append(previous_box[iou_max])
            box=new_box
            state=torch.stack(statels,dim=0)    #update state

            global inverse_rate
            critic_loss=sac.update_policy(lr,ReplayMem,inverse_ReplayMem,inverse_rate)
            
            epoch_critic_loss+=critic_loss.item() 
        
        


            


        

        
    print('------------')
    print('epoch:',str(epoch))
    print(inverse_rate)
    print(gan_rate)
    print('epoch_policy_loss',epoch_critic_loss)
    #print('dis_loss',epoch_unlable_critic_loss)
    print('epoch_reward',epoch_reward/i_count)
    print('epoch_seg_gan_reward',epoch_seg_gan_reward/i_count)
    print('epoch_det_gan_reward',epoch_det_gan_reward/i_count)

    print('epoch_boxreward',epoch_boxreward/i_count)
    print('epoch_segloss',epoch_segloss)
    print('not_update_count:',not_update)
    if i_count!=0:
      print('epoch_iou',epoch_iou/i_count)
      print('epoch_ap',epoch_ap/i_count)
    
    fid = open(directory+'epoch_iou.csv', 'a')
    fid.write(
        '{}, {}, {}, {}\n'.format(epoch, epoch_iou/i_count, epoch_ap/i_count,
                                      epoch_reward/i_count))
    fid.close()
    
    device1=torch.device('cpu')
    criterion=torch.nn.BCEWithLogitsLoss()
    if epoch%40==0:
        dice_=0
        dice_ls=[]
        loss_=0
        iou=0
        iou_ls=[]
        hd_=0
        hd_ls=[]
        count_=0
        kap_=0
        kap_ls=[]
        recall_=0
        recall_ls=[]
        acc_=0
        acc_ls=[]
        
        sen_=0
        sen_ls=[]
        spe_=0
        spe_ls=[]
        
        for idx,(x,label) in enumerate(val_loader):
                count_+=1
                x,label=x.to(device),label.to(device1)
                im=x.clone()
                gt_box=mutibox_from_mask(label)
                
                box=[[20.,20.,100.,100.]]
                state=get_state(x,box)
                

                for i in range(30):
                    action=sac.select_action(state)
                    next_box=execute_mutiaction(box,action)
                    next_state=get_state(x,next_box)
                    box=next_box
                    state=next_state
                
                x=get_state_seg(x,box)

                output=generator(x)
                
                output=torch.squeeze(output,0).to(device1)
                output=get_mask_from_netoutput(output).float().to(device1)
                label=torch.squeeze(label,0).float()
                loss=criterion(output,label)

                pred=output.cpu().clone().detach().numpy()
                pred=torch.from_numpy(pred)
                print(dice.dice_coeff(pred,label))
                loss_+=loss.item()
                
                pred=torch.where(pred>0.5,1,0)
                label=torch.where(label>0.5,1,0)
                dc=dice.dice_coeff(pred,label).item()
                dice_+=dc
                dice_ls.append(dc)
                
                iu=iou_mean(pred,label)
                iou+=iu
                iou_ls.append(iu)
                hdd=hausdorff_distance(pred.clone().cpu().squeeze().numpy(), label.clone().cpu().squeeze().numpy(), distance="manhattan")*0.95
                hd_+=hdd
                hd_ls.append(hdd)
                kp=cohen_kappa_score(pred.cpu().numpy().reshape(-1),label.cpu().numpy().reshape(-1))
                kap_+=kp
                kap_ls.append(kp)
                rc=recall_score(label.cpu().numpy().reshape(-1),pred.cpu().numpy().reshape(-1))
                recall_+=rc
                recall_ls.append(rc)
                ac=accuracy_score(label.cpu().numpy().reshape(-1),pred.cpu().numpy().reshape(-1))
                acc_+=ac
                acc_ls.append(ac)
                
                tn, fp, fn, tp = confusion_matrix(label.cpu().numpy().reshape(-1), pred.cpu().numpy().reshape(-1)).ravel()
                
                sen = tp / (tp + fn)
                sen_+=sen
                sen_ls.append(sen)
                spe = tn / (tn + fp)
                spe_+=spe
                spe_ls.append(spe)
                
                pred=pred.cpu().squeeze(dim=0).numpy()
                label=label.cpu().squeeze(dim=0).numpy()
                
                im1=im.clone().unsqueeze(-1)  #.numpy()
                im2=im.clone().unsqueeze(-1) 
                im3=im.clone().unsqueeze(-1) 
                im= torch.cat([im1,im2,im3],dim=-1)
                x=im.cpu().squeeze().numpy()*255
                x=x.astype(np.uint8)

                
                x=adjust_gamma(x, gamma=1.5)  
                
                
                heat_img = cv2.applyColorMap(x.astype(np.uint8)*255, cv2.COLORMAP_JET)
                heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
                
                a=np.zeros([128,128,3])
                pred=pred.reshape([128,128,1])
                pred=a+pred

                add_img=cv2.addWeighted(heat_img.astype(np.uint8), 0.5, pred.astype(np.uint8)*255, 0.5, 0) 

                cv2.imwrite(str(train_rate).replace('.','')+'_'+str(label_rate).replace('.','')+'/'+str(idx)+'heat.png',add_img)

                fid = open(str(train_rate).replace('.','')+'_'+str(label_rate).replace('.','')+'/_DGARL_detail.csv', 'a')
                fid.write(
        '{}, {}, {}, {}, {},{}, {},{},{},{}\n'.format(str(epoch),idx, dc, hdd,
                                          iu,kp,rc,ac,sen,spe))
                fid.close()

        print('dice',dice_/count_)
        print('iou',iou/count_)
        fid = open(str(train_rate).replace('.','')+'_'+str(label_rate).replace('.','')+'/DGARL-std.csv', 'a')
        fid.write(
    '{}, {}, {}, {},{}, {},{},{}, {}, {},{}, {},{},{}, {},{},{}\n'.format(str(epoch), dice_/count_,np.std(dice_ls), hd_/count_,np.std(hd_ls),
                                        iou/count_,np.std(iou_ls),kap_/count_,np.std(kap_ls),recall_/count_,np.std(recall_ls),acc_/count_,np.std(acc_ls),sen_/count_,np.std(sen_ls),spe_/count_,np.std(spe_ls)))
        fid.close()
        if(dice_/count_>0.84):
          print(1/0)
 
 
     
   


if __name__ == "__main__":
    
    

    sac=SAC()
    generator=ResUNet(image_channel+1,1).to(device)
    target_g=ResUNet(image_channel+1,1).to(device)

    
    discriminator=evalnet.evalNet(2).to(device)
    target_d=evalnet.evalNet(2).to(device)

    ReplayMem=ReplayBuffer(buffer_experience_replay)
    reverse_ReplayMem=ReplayBuffer(buffer_experience_replay)

    data_set=DataSet.liverset('./pancreas/image','./pancreas/label',False)

    train_set,val_set=random_split(dataset=data_set,lengths=[int(len(data_set)*train_rate),len(data_set)-int(len(data_set)*train_rate)],generator=torch.Generator().manual_seed(0))  

    label_set,unlabeled_set=random_split(dataset=train_set,lengths=[int(len(train_set)*label_rate),len(train_set)-int(len(train_set)*label_rate)],generator=torch.Generator().manual_seed(0))

    batch_size=10

    labeled_bs=int(batch_size*label_rate)
    label_loader=DataLoader(label_set,batch_size=int(batch_size*label_rate),shuffle=True,drop_last=True)
    unlabel_loader=DataLoader(unlabeled_set,batch_size=int(batch_size*(1-label_rate)),shuffle=True,drop_last=True)
    val_loader=DataLoader(val_set,batch_size=1,shuffle=False)
    
    ep='160'
    sac.load(ep,directory)
    generator=torch.load(os.path.join(directory,ep+'generator_liver.sp'),map_location=device)
    discriminator=torch.load(os.path.join(directory,ep+'discriminator_liver.sp'),map_location=device)
    hard_update(target_d,discriminator)
    hard_update(target_g,generator)

    dis_d=evalnet.evalNet(2).to(device)
    dis_d=torch.load(os.path.join(directory,ep+'dis_d_liver.sp'),map_location=device)
    dis_d_target=evalnet.evalNet(2).to(device)
    hard_update(dis_d_target,dis_d)
    
    for i in range(0,EPOCHS):
        mixed_training(sac,generator,discriminator,ReplayMem,label_loader,unlabel_loader,val_loader,i,target_d,target_g,dis_d,dis_d_target,reverse_ReplayMem)
        #break
        if i %40==0:
          torch.save(discriminator,os.path.join(directory+str(i)+'discriminator_liver.sp'))
          sac.save(i,directory)
          torch.save(generator,os.path.join(directory+str(i)+'generator_liver.sp'))
          torch.save(dis_d,os.path.join(directory+str(i)+'dis_d_liver.sp'))
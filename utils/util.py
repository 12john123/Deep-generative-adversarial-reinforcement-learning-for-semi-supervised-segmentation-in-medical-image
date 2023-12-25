import os
import torch
import numpy as np
from paramter import *
from torch.autograd import Variable
from torch import autograd
import cv2

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))

flag=0
def execute_mutiaction(last_bbox, action):
    boxls=[]
    for i in range(action.shape[0]):
        box=execute_action(last_bbox[i],action[i])  #,done
        boxls.append(box)
    return boxls 

def execute_action(last_bbox, action):
    """
        actions include 4 kinds:
            [0]. translation vertically        [1]. translation Horizontally
            [2]. vertical scale                [3]. horizon scale
            [5]. stop
    """
    #action=action.detach().cpu().squeeze(dim=0).numpy()
    #action=np.squeeze(action,axis=0)
    center_point_x = int((int(last_bbox[0]) + int(last_bbox[2])) / 2)
    center_point_y = int((int(last_bbox[1]) + int(last_bbox[3])) / 2)

    h = last_bbox[2] - last_bbox[0]
    w = last_bbox[3] - last_bbox[1]
    # print("iou old",iou_old)
    # [0]. translation vertically
    t_v = action[0] * 6
    new_center_x = center_point_x + t_v
    # [1]. translation horizontally
    t_h = action[1] * 6
    new_center_y = center_point_y + t_h

    # [2]. scale vertically
    new_h = int(action[2] * 4) + h
    # new_h = hei_box_def
    # new_w = wid_box_def
    # [3]. scale horizontally
    new_w = int(action[3] * 4) + w      #???
    #new_w = int(action[2] * 4) + w

    if new_w > 48:
        new_w = 48
    if new_w < 4:
        new_w = 4
    if new_h > 48:
        new_h = 48
    if new_h < 4:
        new_h = 4

    lt_x = int(new_center_x - new_w / 2)
    rb_x = int(new_center_x + new_w / 2)

    if lt_x < 0:
        lt_x = 0
        rb_x = new_w
    if rb_x > wid_img_def - 1:
        rb_x = wid_img_def - 1
        lt_x = wid_img_def - 1 - new_w

    lt_y = int(new_center_y - new_h / 2)
    rb_y = int(new_center_y + new_h / 2)

    if lt_y < 0:
        lt_y = 0
        rb_y = new_h
    if rb_y > wid_img_def - 1:
        rb_y = wid_img_def - 1
        lt_y = wid_img_def - 1 - new_h

    curr_bbox = [lt_x, lt_y, rb_x, rb_y]


    return curr_bbox 

def execute_mutiaction_inverse(last_bbox, action):
    boxls=[]
    for i in range(len(last_bbox)):
        for j in range (5):
            box=execute_action_inverse(last_bbox[i],action[i*5+j])  #,done
            boxls.append(box)
    return boxls 

def execute_action_inverse(last_bbox, action):
    """
        actions include 4 kinds:
            [0]. translation vertically        [1]. translation Horizontally
            [2]. vertical scale                [3]. horizon scale
            [5]. stop
    """
    #action=action.detach().cpu().squeeze(dim=0).numpy()
    #action=np.squeeze(action,axis=0)
    center_point_x = int((int(last_bbox[0]) + int(last_bbox[2])) / 2)
    center_point_y = int((int(last_bbox[1]) + int(last_bbox[3])) / 2)

    h = last_bbox[2] - last_bbox[0]
    w = last_bbox[3] - last_bbox[1]
    # print("iou old",iou_old)
    # [0]. translation vertically
    t_v = -action[0] * 6
    new_center_x = center_point_x + t_v
    # [1]. translation horizontally
    t_h = -action[1] * 6
    new_center_y = center_point_y + t_h

    # [2]. scale vertically
    new_h = int(-action[2] * 4) + h
    # new_h = hei_box_def
    # new_w = wid_box_def
    # [3]. scale horizontally
    new_w = int(-action[3] * 4) + w      #???
    #new_w = int(action[2] * 4) + w

    if new_w > 48:
        new_w = 48
    if new_w < 4:
        new_w = 4
    if new_h > 48:
        new_h = 48
    if new_h < 4:
        new_h = 4

    lt_x = int(new_center_x - new_w / 2)
    rb_x = int(new_center_x + new_w / 2)

    if lt_x < 0:
        lt_x = 0
        rb_x = new_w
    if rb_x > wid_img_def - 1:
        rb_x = wid_img_def - 1
        lt_x = wid_img_def - 1 - new_w

    lt_y = int(new_center_y - new_h / 2)
    rb_y = int(new_center_y + new_h / 2)

    if lt_y < 0:
        lt_y = 0
        rb_y = new_h
    if rb_y > wid_img_def - 1:
        rb_y = wid_img_def - 1
        lt_y = wid_img_def - 1 - new_h

    curr_bbox = [lt_x, lt_y, rb_x, rb_y]


    return curr_bbox



def get_mutireward(old,new):
    rewardls=[]
    for i in range(old.shape[0]): 
        reward=get_reward(old[i],new[i])
        rewardls.append(reward)
    reward=torch.stack(rewardls,dim=0)
    return reward

def get_reward(old,new):

    if new <0.5:
        if new-old>0.1:
            reward=torch.tensor([1]).to(device)*torch.exp(new*10).to(device)/120
        elif new-old<-0.1:
            reward=torch.tensor([-1]).to(device)*torch.exp(new*10).to(device)/120
        else:
            reward=(new-old).to(device)*10*torch.exp(new*10)/120 
    else:
        if new-old>0.1:
            reward=torch.tensor([1]).to(device)*torch.exp(new*10).to(device)/120+2
        elif new-old<-0.1:
            reward= torch.tensor([-1]).to(device)*torch.exp(new*10).to(device)/120+2
        elif new-old>-0.1 and new-old<0:
            reward=torch.tensor([0]).to(device)    #0     #(new-old).to(device)*10*torch.exp(new*10)/120
        else: 
            reward=(new-old).to(device)*10*torch.exp(new*10)/120+2

    return reward

def get_mask_from_netoutput(output):
    mask=output.clone().detach().cpu().numpy()
    mask=np.where(mask>0.5,1,0)
    mask=torch.from_numpy(mask)
    mask=mask.to(device)
    return mask



def get_dis(last_bbox, target_bbox):
    dis = np.linalg.norm(np.subtract(last_bbox, target_bbox))
    return dis

def get_iou(last_bbox, target_bbox):
    pred_old = np.zeros([wid_img_def, hei_img_def])
    pred_old[int(last_bbox[0]):int(last_bbox[2]), int(last_bbox[1]):int(last_bbox[3])] = 1

    target = np.zeros([wid_img_def, hei_img_def])
    # print("last bbox", last_bbox)
    # print("target bbox", target_bbox)
    target[int(target_bbox[0]):int(target_bbox[2]), int(target_bbox[1]):int(target_bbox[3])] = 1

    pred_old = np.int64(pred_old)
    target = np.int64(target)
    map_and = np.bitwise_and(pred_old, target)
    map_or = np.bitwise_or(pred_old, target)
    if np.sum(map_or) != 0:
        iou_old = np.sum(map_and) * 1.0 / np.sum(map_or)
    else:
        iou_old = 0

    return iou_old

def get_ap(last_bbox, target_bbox):
    pred_old = np.zeros([wid_img_def, hei_img_def])
    pred_old[int(last_bbox[0]):int(last_bbox[2]), int(last_bbox[1]):int(last_bbox[3])] = 1

    target = np.zeros([wid_img_def, hei_img_def])
    # print("last bbox", last_bbox)
    # print("target bbox", target_bbox)
    target[int(target_bbox[0]):int(target_bbox[2]), int(target_bbox[1]):int(target_bbox[3])] = 1

    pred_old = np.int64(pred_old)
    target = np.int64(target)
    map_and = np.bitwise_and(pred_old, target)
    # map_or = np.bitwise_or(pred_old, target)
    if np.sum(target) != 0:
        iou_old = np.sum(map_and) * 1.0 / np.sum(target)
    else:
        iou_old = 0

    return iou_old

def get_recall(last_bbox, target_bbox):
    pred_old = np.zeros([wid_img_def, hei_img_def])
    pred_old[int(last_bbox[0]):int(last_bbox[2]), int(last_bbox[1]):int(last_bbox[3])] = 1

    target = np.zeros([wid_img_def, hei_img_def])
    # print("last bbox", last_bbox)
    # print("target bbox", target_bbox)
    target[int(target_bbox[0]):int(target_bbox[2]), int(target_bbox[1]):int(target_bbox[3])] = 1

    pred_old = np.int64(pred_old)
    target = np.int64(target)
    map_and = np.bitwise_and(pred_old, target)
    # map_or = np.bitwise_or(pred_old, target)
    if np.sum(target) != 0:
        recall = np.sum(map_and) * 1.0 / np.sum(pred_old)
    else:
        recall = 0

    return recall

def get_state(img, curr_bbox):
    input_bbox = torch.zeros(img.shape[0], 1, wid_img_def, hei_img_def)

    for i in range(img.shape[0]):
        input_bbox[i, 0, int(curr_bbox[i][0]):int(curr_bbox[i][2]), int(curr_bbox[i][1]):int(curr_bbox[i][3])] = 1
    
    input_bbox = input_bbox.float()
    input_img=img.clone().detach()
    input = torch.cat((input_img, input_bbox.to(device)), 1)
    return input

def get_state_seg(img, curr_bbox):
    input_bbox = torch.zeros(img.shape[0], 1, wid_img_def, hei_img_def)

    for i in range(img.shape[0]):
        input_bbox[i, 0, int(curr_bbox[i][0]):int(curr_bbox[i][2]), int(curr_bbox[i][1]):int(curr_bbox[i][3])] = 1
    
    input_bbox = input_bbox.float()
    input_img=img.clone().detach()
    input = torch.cat((input_img*input_bbox.to(device).detach(), input_bbox.to(device)), 1)
    return input

def get_state_previous(img, curr_bbox):
    input_bbox = torch.zeros(len(curr_bbox), 1, wid_img_def, hei_img_def)

    imgls=[]
    for i in range(len(img)):
        for j in range(5):
            input_bbox[i*5+j, 0, int(curr_bbox[i*5+j][0]):int(curr_bbox[i*5+j][2]), int(curr_bbox[i*5+j][1]):int(curr_bbox[i*5+j][3])] = 1
            imgls.append(img[i].clone().detach())
    input_bbox = input_bbox.float()
    #input_img=img.clone().detach()
    input_img=torch.stack(imgls,dim=0)

    input = torch.cat((input_img, input_bbox.to(device)), 1)
    return input

def get_dis_reward(gt_box,bbox):
    #gt_box=mutibox_from_mask(gt)

    #[l_x, l_y, r_x, r_y]
    disls=[]
    for i in range(len(gt_box)):
        lx_t=np.max([gt_box[i][0],127-gt_box[i][0]])
        lx_dis=np.absolute(gt_box[i][0]-bbox[i][0])
        lx=lx_dis/lx_t

        ly_t=np.max([gt_box[i][1],127-gt_box[i][1]])
        ly_dis=np.absolute(gt_box[i][1]-bbox[i][1])
        ly=ly_dis/ly_t

        rx_t=np.max([gt_box[i][2],127-gt_box[i][2]])
        rx_dis=np.absolute(gt_box[i][2]-bbox[i][2])
        rx=rx_dis/rx_t

        ry_t=np.max([gt_box[i][3],127-gt_box[i][3]])
        ry_dis=np.absolute(gt_box[i][3]-bbox[i][3])
        ry=ry_dis/ry_t


        score=1-(lx+ly+rx+ry)/4
        disls.append(torch.from_numpy(np.array(np.float32(score))).to(device))
    
    return torch.stack(disls,dim=0)








def mutibox_from_mask(mutimask):
    boxls=[]
    for i in range(mutimask.shape[0]):
        boxls.append(box_from_mask(mutimask[i].cpu().numpy()))
    return boxls

def box_from_mask(mask):
        mask=mask.reshape((mask.shape[-2],mask.shape[-1]))
        if np.max(mask)==0:
            return [mask.shape[-2]/2, mask.shape[-2]/2, mask.shape[-2]/2+12, mask.shape[-2]/2+12]
        mask = mask / np.max(mask)
        [x, y] = np.where(mask == 1)
        l_x = np.min(x)
        r_x = np.max(x)
        l_y = np.min(y)
        r_y = np.max(y)

        return [l_x, l_y, r_x, r_y]
        
        center_x = np.round((l_x + r_x)/2)
        center_y = np.round((l_y + r_y) / 2)

        side_length = np.maximum(r_x - l_x, r_y - l_y)
        half_side_length = np.round(side_length/2)

        if center_x - half_side_length < 0:
            half_side_length = center_x
        if center_y - half_side_length < 0:
            half_side_length = center_y

        if center_x + half_side_length > wid_img_def - 1:
            half_side_length = wid_img_def - 1 - center_x
        if center_y - half_side_length > wid_img_def - 1:
            half_side_length = wid_img_def - 1 - center_y

        l_x = center_x - half_side_length
        r_x = center_x + half_side_length
        l_y = center_y - half_side_length
        r_y = center_y + half_side_length

        return [l_x, l_y, r_x, r_y]

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, device=device):
    return torch.tensor(ndarray, dtype=torch.float, device=device)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def cal_gradient_penalty(discrimnator,real_data, fake_data, batch_size):
        #print(real_data.size())
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
        alpha = alpha.view(batch_size, 2, wid_img_def, wid_img_def)
        alpha = alpha.to(device)
        fake_data = fake_data.view(batch_size, 2, wid_img_def, wid_img_def)
        interpolates = Variable(alpha * real_data.data + ((1 - alpha) * fake_data.data), requires_grad=True)
        disc_interpolates = discrimnator(interpolates)
        gradients = autograd.grad(disc_interpolates, interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty    
def draw_box_on_img(img, last_bbox, target_bbox):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #print(last_bbox[1])
    last_bbox=last_bbox[0]
    target_bbox=target_bbox[0]
    img[last_bbox[0], last_bbox[1]: last_bbox[3], 1] = 255
    img[last_bbox[2], last_bbox[1]: last_bbox[3], 1] = 255
    img[last_bbox[0]: last_bbox[2], last_bbox[1], 1] = 255
    img[last_bbox[0]: last_bbox[2], last_bbox[3], 1] = 255

    img[int(target_bbox[0]), int(target_bbox[1]): int(target_bbox[3]), 0] = 255
    img[int(target_bbox[2]), int(target_bbox[1]): int(target_bbox[3]), 0] = 255
    img[int(target_bbox[0]): int(target_bbox[2]), int(target_bbox[1]), 0] = 255
    img[int(target_bbox[0]): int(target_bbox[2]), int(target_bbox[3]), 0] = 255
    return img
def iou_mean(pred, target, n_classes = 1):
#n_classes ：the number of classes in your dataset,not including background
# for mask and ground-truth label, not probability map
  ious = []
  iousSum = 0
  #pred = torch.from_numpy(pred)
  pred = pred.reshape(-1)
  #target = np.array(target)
  #target = torch.from_numpy(target)
  target = target.reshape(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
      iousSum += float(intersection) / float(max(union, 1))
  return iousSum/n_classes


def box_reward(dis_old, dis_new, iou_old, iou_new, ap_old, ap_new):
    overlap_new = (iou_new * 0.8 + ap_new * 0.2)    #

    if overlap_new <= 0:
        reward=0
        # if dis_old - dis_new < 0:
        #     reward = dis_old - dis_new
        # else:
        #     reward = np.exp(-dis_new / 100) * 4
    else:
        # if dis_old - dis_new < 0:
        #     reward = 0
        # else:
        reward = np.exp(overlap_new*3)       #(np.exp(-dis_new / 100)) * 4 +

    return reward

def gan_reward(gan_score):
    
    reward = np.exp(gan_score.cpu().numpy()*3)

    return reward


def get_ganreward(gan_scorels):
    rewardls=[]
    for i in range(gan_scorels.size(0)):
            r=gan_reward(gan_scorels[i])
            r=torch.from_numpy(np.array(np.float32(r)))
            rewardls.append(r)
    reward=torch.stack(rewardls,dim=0)
    return reward

def get_ganreward_previous(gan_scorels):
    rewardls=[]
    for i in range(gan_scorels.size(0)):
            r=gan_reward(gan_scorels[i])
            for j in range(5):
                r=torch.from_numpy(np.array(np.float32(r)))
                rewardls.append(r)
    reward=torch.stack(rewardls,dim=0)
    return reward



def get_boxreward(target, last,curr):
    rewardls=[]
    iouls=[]
    apls=[]
    for i in range(len(target)):
        target_bbox=target[i]
        last_bbox=last[i]
        curr_bbox=curr[i]
        iou_old = get_iou(last_bbox, target_bbox)
        iou_new = get_iou(curr_bbox, target_bbox)
        ap_old = get_ap(last_bbox, target_bbox)
        ap_new = get_ap(curr_bbox, target_bbox)
        dis_old = get_dis(last_bbox, target_bbox)
        dis_new = get_dis(curr_bbox, target_bbox)
        r=box_reward(dis_old, dis_new, iou_old, iou_new, ap_old, ap_new)
        r=torch.from_numpy(np.array(np.float32(r)))

        rewardls.append(r)
        iouls.append(iou_new)
        apls.append(ap_new)
    reward=torch.stack(rewardls,dim=0)
    return reward,np.mean(iouls),np.mean(apls),apls,iouls

def get_boxreward_previous(target, last,curr):
    rewardls=[]
    iouls=[]
    apls=[]
    for i in range(len(target)):
        for j in range(5):
            target_bbox=target[i]
            last_bbox=last[i*5+j]
            curr_bbox=curr[i]
            iou_old = get_iou(last_bbox, target_bbox)
            iou_new = get_iou(curr_bbox, target_bbox)
            ap_old = get_ap(last_bbox, target_bbox)
            ap_new = get_ap(curr_bbox, target_bbox)
            dis_old = get_dis(last_bbox, target_bbox)
            dis_new = get_dis(curr_bbox, target_bbox)
            r=box_reward(dis_old, dis_new, iou_old, iou_new, ap_old, ap_new)
            r=torch.from_numpy(np.array(np.float32(r)))

            rewardls.append(r)
            iouls.append(iou_new)
            apls.append(ap_new)
    reward=torch.stack(rewardls,dim=0)
    return reward,np.mean(iouls),np.mean(apls),apls,iouls

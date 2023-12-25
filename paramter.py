import os

import torch

ExpDir = './data/merge/'
ResDir = './net_save/'

net_save_path='./net_save/'
test_save_path='./test_result/'
multi_GPUs = False

use_cuda = torch.cuda.is_available()
USE_CUDA = torch.cuda.is_available()
device = torch.device(  "cpu") #"cuda0" if use_cuda else

number_of_steps = 5
buffer_experience_replay = 2000


training_num_per_iteration=6

discount=0.99

rl_batch_size = 15
batch_size = 15


image_channel=1

image_w=128
image_h=128

DATASET_PATH = ExpDir
EPOCHS = 1000

EPS_START = 0.95
EPS_END = 0.006

decay = 0.95

#####
## learning rate
learning_rate = 1e-4


gamma=0.99
mean_lambda=1e-3
std_lambda=1e-3
z_lambda=0.0
tau=1e-3

WEIGHT_DECAY = 0.0001
LR_DECAY = 0.8
CUDNN = True
seed = 0
epsilon =20000

directory='./net_save/'

# for dataset.py

img_dir = ExpDir + 'train/'

# for experiment.py
Interval_save_model = 4000
# Interval_save_model = 4
weight_save_path = ResDir + 'model/'
if not os.path.exists(weight_save_path):
    os.makedirs(weight_save_path)

#for utils.py
# period = 50000
# patience = 10000
# TH_meaning = 0.01
history_dir = ResDir + 'doc/'
if not os.path.exists(history_dir):
    os.makedirs(history_dir)
training_csv_path = os.path.join(history_dir, 'training.csv')
teacher_val_csv_path = os.path.join(history_dir, 'val_teacher.csv')
student_val_csv_path = os.path.join(history_dir, 'val_student.csv')


wid_img_def = 128
hei_img_def = 128
#hei_box_def = 12
#wid_box_def = 12

ou_theta = 0.05
ou_sigma = 0.10
ou_mu = 0.0
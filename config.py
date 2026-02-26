import os
import torch


work_path = ''#../MS-edgecast
mod = ''

# data path
train_data_root = ''
test_data_root = ''
valid_data_root = ''

# checkpoint path
ckpt_dir = work_path+'/checkpoint{}/'.format(mod)
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)


use_gpu = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 2
num_workers = 0
print_freq = 100

max_epoch = 300
lr = 1e-4
momentum = 0.9
weight_decay = 0.004


out_len = 12
in_len = 6
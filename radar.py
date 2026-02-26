# @Author:wlt
# @File: radar.py
# 2023/10/12 21:06


from doctest import debug
import os
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import config as cfg
import matplotlib.colors as colors
import random
from torch import nn
import h5py

class radarDataset(data.Dataset):
    def __init__(self, root):
        self.root = root
        with open(root, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]



    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        with h5py.File(self.samples[idx], 'r') as f:
            self.train_data = f['data'][()] / 80
            self.train_data = np.clip(self.train_data, 0.0, 1.0)
            data = self.train_data[0:6, :, :,:]
            lable= self.train_data[6:18, :, :,:]
        f.close()
        return torch.from_numpy(lable).float(),torch.from_numpy(data).float()
    


class radar_evo_Dataset(data.Dataset):
    #framewise
    def __init__(self, root,t=0,rd = False):
        self.rd = rd
        self.root = root
        self.t = t
        with open(root, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        #if self.rd == True:
           
        with h5py.File(self.samples[idx], 'r') as f:
            self.train_data = f['data'][()]
            if self.rd == True:
                self.t = random.randint(0,3)
            label = self.train_data[self.t*3+18:self.t*3+18+3,0,:,:]


            data = np.concatenate([self.train_data[:6,:,:,:],self.train_data[6+self.t*3:6+self.t*3+3,:,:,:]],axis=0).squeeze()
        f.close()
        return torch.from_numpy(label).float(),torch.from_numpy(data).float(),self.t



class radarDataset_evo(data.Dataset):
    def __init__(self, root):
        self.root = root
        with open(root, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]



    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        with h5py.File(self.samples[idx], 'r') as f:
            self.train_data = f['data'][()]
            label = self.train_data[18:,:,:,:]
            data = self.train_data[:18,:,:,:]
        f.close()
        return torch.from_numpy(label).float(),torch.from_numpy(data).float()

# dataloader file for training the CVAE and ACVAE
# (c) MERL 2019, Yingheng Tang

from __future__ import print_function, division
import scipy.io as sio
import scipy.stats
import h5py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils





class HVDataset(Dataset):
    """dataset loader"""

    def __init__(self, dir, HV_input, label_input, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = loadmat(dir)
        self.inputt = self.x.get(HV_input)
        self.inputt2 = self.x.get(label_input)
        self.inputt = self.inputt.astype(np.float32)
        self.inputt2 = self.inputt2.astype(np.float32)
        # self.transform = transform
    def __len__(self):
        return len(self.inputt)

    def __getitem__(self,index):
        HV = self.inputt[index]
        label = self.inputt2[index]
        return HV, label
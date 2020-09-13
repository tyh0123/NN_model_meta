
from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats
import h5py
import pandas as pd
from dataset_NN import HVDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable




parser = argparse.ArgumentParser(description='NN_code')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


#comment out this line to enable the random seed
# torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


##load the file
cwd = os.getcwd()
filename = 'data_radius.mat'   ##load the .mat file to the current directory and change the name here
dir = os.path.join(cwd,filename)
HV_input = 'radius'                         ##variable name for the HV in the .mat.file
label_input = 'T'                      ##variable name for the label in the .mat file

train = HVDataset(dir, HV_input, label_input, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train,batch_size=args.batch_size, shuffle=True, **kwargs)




##NN network
class NN_network(nn.Module):
    def __init__(self):
        super(NN_network, self).__init__()

        # encoder graph
        self.layer1 = nn.Sequential(
            nn.Linear(4, 16),
            nn.Dropout(0.2),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Linear(16, 128),
            nn.Dropout(0.2),
            nn.ReLU())

        self.layer3 = nn.Sequential(
            nn.Linear(128, 1024),
            nn.Dropout(0.4),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(0.4),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU())

        self.layer6 = nn.Sequential(
            nn.Linear(512, 200),
            nn.Sigmoid())





    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out






model = NN_network().to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)



##loss function for the adversarial block
def loss_function(c_bar, c):
    loss = F.mse_loss(c_bar,c,reduction='sum')
    return loss


###training
def train(epoch):
    loss_record = []
    adv_los_record = []
    kld_record = []
    bce_record = []
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = torch.unsqueeze(data, 1)
        data = data.to(device)
        labels = labels.to(device)
        outp = model(data)
        outp = outp.view(-1,200)
        optimizer.zero_grad()
        loss = loss_function(outp, labels)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        loss_record = np.append(loss_record,loss.item()/len(data))
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader),
            loss.item()))

    return loss_record





def plot_loss(total):
    plt.plot(l1, label='TOTAL_LOSS')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    l1 = []     ##total loss record
    for epoch in range(1, args.epochs + 1):
        print(epoch%3)
        loss_t = train(epoch)
        l1 = np.append(l1, loss_t)
    plot_loss(l1)
    torch.save(model, 'NN_model.pkl')         ##save the model

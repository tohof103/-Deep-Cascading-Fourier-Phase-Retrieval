# Define imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import random

from fourier import fft2WoSq, magnitude, phase, fft2
from pics import greypic, blackpic, meanpic, DFPRsaveToPDF
from tests import DFPRtest
from models import DFSigmoidNet, ConvNet
from loader import load
torch.manual_seed(17)
random.seed(0)


# Define initialization

def init(new, Net, NUM_Of_Nets, netsize, device, lr, targetsize, dataset):
    '''
    Initializes/loads networks and optimizer
    Parameters:
    -----------
        new: boolean
            Create new network or use previous if existing
        Net: nn.module class
            Class of network to be used
        NUM_Of_Nets: int
            Number of networks
        netsize: int array
            Sizes of fully connected layers of networks
        device: torch.device
            device to initialize the networks to
        lr: float
            Learning rate for optimizer
        targetsize: int array
            Size of the target images as Array (channels, height, width)
        dataset: string
            Name of dataset to load old network
    Returns:
    --------
        nets: nn.module array
        optimizer: torch.optim
    '''
    nets = []
    optimizer = []
    for i in range(NUM_Of_Nets):
        if os.path.isfile('Nets/{}/DFPR{}{}.pt'.format(dataset, netsize, i)):
            if(new):
                os.remove('Nets/{}/DFPR{}{}.pt'.format(dataset, netsize, i))
                print('deleted old Net')
                nets.append(Net(imsize=[targetsize[0] * 2, targetsize[1], targetsize[2]],
                                outsize=targetsize, h = netsize))
            else:
                print('loaded previous Net')
                nets.append(torch.load('Nets/{}/DFPR{}{}.pt'.format(dataset, netsize, i)))
        else:
            nets.append(Net(imsize=[targetsize[0] * 2, targetsize[1], targetsize[2]],
                            outsize=targetsize, h = netsize))
        
        nets[i] = nets[i].to(device)        
        optimizer.append(optim.Adam(nets[i].parameters(), lr = lr))    
    return nets, optimizer


# Define training

def train_epoch(epoch, NUM_Of_Nets, device, startpic, nets, optimizer, losses, train_data):
    '''
    Trains networks for one epoch
    Parameters:
    -----------
        epoch: int
            Current epoch number
        NUM_Of_Nets: int
            Number of networks
        device: torch.device
            device to train on
        startpic: float array
            Start image to feed to first network
        nets: nn.module array
            Array of networks to train
        optimizer: torch.optim array
            Array of optimizer for training
        losses: (loss-)function array
            Array of losses used to train each network
        train_data: dataloader
            Dataloader of training data        
    Returns:
    --------
        -
    '''
    tot_loss = 0
    for net in nets:
        net.train()
            
    for target in train_data:
        target = target.to(device)
        x = startpic.to(device)
        measurement = magnitude(fft2WoSq(target)).to(device)
        
        for step in range(NUM_Of_Nets):
            x = x.detach()            
            fourier_t = fft2WoSq(x)            
            phase_x = phase(fourier_t)
            delta_y = magnitude(fourier_t) - measurement
            data = torch.cat([delta_y, phase_x], 1).detach()
            optimizer[step].zero_grad()
            out = nets[step](data)
                
            out = out * ((0.5-(step%2))*2)    #alternating addition and substraction        
            x = torch.clamp(x + out, 0, 1)
            
            criterion = losses[step]            
            loss = criterion(x, target)            
            loss.backward()            
            optimizer[step].step()
            
        tot_loss = tot_loss + loss.item()
    print('Epoche: {:3.0f} | Loss: {:.6f}'.format(epoch, tot_loss/len(train_data)))

def train(NUM_Of_Nets, device, startpic, nets, optimizer, losses, data, num_epochs):
    '''
    Trains networks for given number of epochs
    Parameters:
    -----------
        NUM_Of_Nets: int
            Number of networks
        device: torch.device
            device to train on
        startpic: float array
            Start image to feed to first network
        nets: nn.module array
            Array of networks to train
        optimizer: torch.optim array
            Array of optimizer for training
        losses: (loss-)function array
            Array of losses used to train each network
        data: dataloader array
            Dataloader of dataset data 
        num_epochs: int
            Number of epochs to train
    Returns:
    --------
        -
    '''
    train_data = data['train']
    val_data = data['val']
    print("========================================")
    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch, NUM_Of_Nets, device, startpic, nets, optimizer, losses, train_data)
        if epoch % 10 == 0:
            DFPRtest(nets, val_data, startpic, NUM_Of_Nets, device, True, False)


# Set hyperparameters

new = False
save = False
dataset = 'celeba'
Net = ConvNet     #ConvNet for CelebA, DFSigmoidNet for (fashion-)MNIST
NUM_Of_Nets = 5
device = torch.device("cuda:1")
netsize = 1700
lr = 0.0001
startpicgen = greypic
losses = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.L1Loss(), nn.L1Loss()] 


# Initializing, training and possible saving

#data, targetsize = load(name = dataset)                            #For (fashion-)MNIST
data, targetsize = load(name = 'celeba', path="CelebA/CelebA.h5")   #For CelebA

nets, optimizer = init(new, Net, NUM_Of_Nets, netsize, device, lr, targetsize, dataset)
startpic = startpicgen(targetsize[0], targetsize[1])

start_proc = time.process_time() 
train(NUM_Of_Nets, device, startpic, nets, optimizer, losses, data, num_epochs = 50)
ende_proc = time.process_time()
print('Systemzeit: {:5.3f}s'.format(ende_proc-start_proc))

if save:
    for i in range(NUM_Of_Nets):
        torch.save(nets[i], 'Nets/{}/DFPR{}{}.pt'.format(dataset, netsize, i))
        print('DFPR{}{} saved'.format(netsize, i))


# Tests and print to PDF

test_data = data['test']
val_data = data['val']
DFPRtest(nets, test_data, startpic, NUM_Of_Nets, device, True, False)
#DFPRsaveToPDF(nets, test_data, startpic, NUM_Of_Nets, device)


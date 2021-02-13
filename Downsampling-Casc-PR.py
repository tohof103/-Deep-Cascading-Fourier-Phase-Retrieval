# Define imports

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import random

from fourier import fft2WoSq, magnitude, squaredmagnitude, fft2
from pics import downsamplingsaveToPDF, plot_grid
from tests import downsampling_test
from models import DCSigmoidNet, ConvNet
from loader import load
torch.manual_seed(17)
random.seed(0)


# Define initialization

def init(new, Net, NUM_Of_Nets, im_size, netsizes, channels, device, lr, dataset):
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
        im_size: int array
            Array of reconstruction sizes
        netsizes: int array
            Sizes of fully connected layers of networks
        channels: int
            Number of color channels
        device: torch.device
            Device to initialize the networks to
        lr: int
            Learning rate for optimizer
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
        if os.path.isfile('Nets/{}/CasPR{}{}.pt'.format(dataset, netsizes[i], i)):
            if(new):
                os.remove('Nets/{}/CasPR{}{}.pt'.format(dataset, netsizes[i], i))
                print('deleted old Net')
                if i == 0:
                    nets.append(Net(imsize=(channels, im_size[-1], im_size[-1]),
                                    outsize=(channels, im_size[i], im_size[i]),
                                    h = netsizes[i]))
                else:
                    nets.append(Net(imsize=(channels*2, im_size[-1], im_size[-1]),
                                    outsize=(channels, im_size[i], im_size[i]),
                                    h = netsizes[i]))
            else:
                print('loaded previous Net')
                nets.append(torch.load('Nets/{}/CasPR{}{}.pt'.format(dataset, netsizes[i], i)))
        else:
            if i == 0:
                nets.append(Net(imsize=(channels, im_size[-1], im_size[-1]),
                                outsize=(channels, im_size[i], im_size[i]),
                                h = netsizes[i]))
            else:
                nets.append(Net(imsize=(channels*2, im_size[-1], im_size[-1]),
                                outsize=(channels, im_size[i], im_size[i]),
                                h = netsizes[i]))
                
        nets[i] = nets[i].to(device)        
        optimizer.append(optim.Adam(nets[i].parameters(), lr = lr))        
    return nets, optimizer


# Define training

def train_epoch(epoch, NUM_Of_Nets, device, im_size, beta, nets, optimizer, losses, train_data):
    '''
    Trains networks for one epoch
    Parameters:
    -----------
        epoch: int
            Current epoch number
        NUM_Of_Nets: int
            Number of networks
        device: torch.device
            Device to train on            
        im_size: int array
            Array of reconstruction sizes
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
    for net in nets:
        net.train()
    
    tot_loss = 0
    reg_loss = 0
        
    for target in train_data:
        target = target.to(device)
        
        for step in range(NUM_Of_Nets):
            stage_target = F.interpolate(target, size=(im_size[step],im_size[step]))
            data = magnitude(fft2WoSq(target)).to(device)
            if step > 0:
                out = F.interpolate(out, size=(im_size[-1], im_size[-1]))
                data = torch.cat([data, out], 1)
                
            optimizer[step].zero_grad()    
            data = data.detach()
            out = nets[step](data)
            
            criterion = losses[step]
            loss = criterion(out, stage_target)
            
            if beta > 0:
                mags = squaredmagnitude(fft2(stage_target))
                mags = mags.detach()
                reg = beta*torch.mean((squaredmagnitude(fft2(out))-mags)**2)
                loss = loss + reg
                
            loss.backward()
            optimizer[step].step()
    
        tot_loss = tot_loss + loss.item()
    print('Epoche: {:3.0f} | Loss: {:.6f}'.format(epoch, tot_loss/len(train_data)))

def train(NUM_Of_Nets, device, im_size, beta, nets, optimizer, losses, data, num_epochs):
    '''
    Trains networks for given number of epochs
    Parameters:
    -----------
        NUM_Of_Nets: int
            Number of networks
        device: torch.device
            Device to train on
        im_size: int array
            Array of reconstruction sizes
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
    print("=======================================")
    for epoch in range(1, num_epochs + 1):
        train_epoch(epoch, NUM_Of_Nets, device, im_size, beta, nets, optimizer, losses, train_data)
        if epoch % 10 == 0:
            downsampling_test(nets, val_data, NUM_Of_Nets, device, im_size, True, False)


# Set hyperparameters

new = False
save = False
dataset = 'celeba'
Net = ConvNet     #ConvNet for CelebA, DCSigmoidNet for (fashion-)MNIST
Num_Of_Nets = 5
device = torch.device("cuda:1")
netsizes = [1100, 1300, 1500, 1800, 1800]
lr = 0.0001
#im_size = [7,14,21,28,28]    #For (fashion-)MNIST
im_size = [16,32,48,64,64]   #For CelebA
beta = 0#2e-4
losses = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.L1Loss()]   


# Initializing, training and possible saving

#data, targetsize = load(name = dataset)                            #For (fashion-)MNIST
data, targetsize = load(name = 'celeba', path="CelebA/CelebA.h5")   #For CelebA
nets, optimizer = init(new, Net, Num_Of_Nets, im_size, netsizes, targetsize[0], device, lr, dataset)

start_proc = time.process_time()
train(Num_Of_Nets, device, im_size, beta, nets, optimizer, losses, data, num_epochs = 50)    
ende_proc = time.process_time()
print('Systemzeit: {:5.3f}s'.format(ende_proc-start_proc))

if save:
    for i in range(Num_Of_Nets):
        torch.save(nets[i], 'Nets/{}/CasPR{}{}.pt'.format(dataset, netsizes[i], i))
        print('CasPR{}{} saved'.format(netsizes[i], i))


# Tests and print to PDF

test_data = data['test']
val_data = data['val']
downsampling_test(nets, test_data, Num_Of_Nets, device, im_size, pics = True, save = False)
#downsamplingsaveToPDF(nets, test_data, Num_Of_Nets, device, im_size)


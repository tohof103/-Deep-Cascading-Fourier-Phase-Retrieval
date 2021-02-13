import torch
import torch.nn as nn
import numpy as np


class DFSigmoidNet(nn.Module):
    '''
    Neural network for DFPR(-woPh) inheriting from nn.module
    Consists of 5 FC layer with batchnorm and ReLu in between
    Additional Dropout layer after first FC layer
    Parameters:
    -----------
        imsize (optional): int array
            size of input image (colors, h, w), default: (1, 28, 28)
        outsize (optional): int array
            size of output image (colors, h, w), default: (1, 28, 28)
        h (optional): int
            Size of FC layer, default: 1024
    '''
    def __init__(self, imsize=(1, 28, 28), outsize=(1,28,28), h=1024):
        super(DFSigmoidNet, self).__init__()
        print("DFSigmoidNet: 5.1 LDLLLL, d=0.3, size: ", h)
        self.imsize = imsize
        if outsize is None:
            self.outsize = imsize
        else:
            self.outsize = outsize
                
        self.layers = nn.Sequential(
            nn.Linear(imsize[0] * imsize[1] * imsize[2], h),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),            
            nn.ReLU(),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),            
            nn.ReLU(),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, self.outsize[0] * self.outsize[1] * self.outsize[2]),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        '''Forward pass through network'''
        N = x.shape[0]
        out = self.layers(x.view(N, -1))
        return out.view(N, *self.outsize)


class DCSigmoidNet(nn.Module):
    '''
    Neural network for DCPR inheriting from nn.module
    Consists of 5 FC layer with batchnorm and ReLu in between
    Two additional Dropout layer after first and after third FC layer 
    Parameters:
    -----------
        imsize (optional): int array
            size of input image (colors, h, w), default: (1, 28, 28)
        outsize (optional): int array
            size of output image (colors, h, w), default: (1, 28, 28)
        h (optional): int
            Size of FC layer, default: 1024
    '''    
    def __init__(self, imsize=(1, 28, 28), outsize=(1,28,28), h=1024):
        super(DCSigmoidNet, self).__init__()
        print("DCSigmoidNet: 5.2 LDLLDLL, d=0.25, size: ", h)
        self.imsize = imsize
        if outsize is None:
            self.outsize = imsize
        else:
            self.outsize = outsize
                
        self.layers = nn.Sequential(
            nn.Linear(imsize[0] * imsize[1] * imsize[2], h),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),            
            nn.ReLU(),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),            
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(h,h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, self.outsize[0] * self.outsize[1] * self.outsize[2]),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        '''Forward pass through network'''   
        N = x.shape[0]
        out = self.layers(x.view(N, -1))
        return out.view(N, *self.outsize)
    

class KernModule(nn.Module):
    '''
    Neural network called from ConvNet inheriting from nn.module
    Consists of 2 FC layer with batchnorm and ReLu in between
    Additional Dropout layer after first FC layer 
    Parameters:
    -----------
        h: int
            Size of FC layer,
            
    '''
    def __init__(self, h):
        super(KernModule, self).__init__()
        self.h = h
        self.layers = nn.Sequential(
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU()
        )
        
    def forward(self, x):
        '''Forward pass through network'''
        shp = x.shape
        return self.layers(x.view(shp[0], self.h)).view(*shp)

#Used with permission from Alexander Oberstraß and Tobias Uelwer    
class ConvNet(nn.Module):
    '''
    Neural network for CelebA dataset inheriting from nn.module
    Consists of convolutional layer with maxpooling, batchnorm and ReLu in between
    After FC-KernModule transposed convolutional layer and upsampling
    Parameters:
    -----------
        imsize (optional): int array
            size of input image (colors, h, w), default: (1, 28, 28)
        outsize (optional): int array
            size of output image (colors, h, w), default: (1, 28, 28)
        h (optional): int
            For generalization during initialization
        s (optional): int
            Defines kernel sizes, default: 32
    '''

    def __init__(self, imsize=(1, 28, 28), outsize=None, h = 1024, s=32):
        super(ConvNet, self).__init__()
        pow_pad = (2 ** (int(np.ceil(np.log2(imsize[-2])))) - imsize[-2],
                   2 ** (int(np.ceil(np.log2(imsize[-1])))) - imsize[-1])
        kern_size = 4 * ((imsize[1] + pow_pad[0]) // 16) * ((imsize[2] + pow_pad[1]) // 16) * s
        print("Additional padding to fit 2 exp:", pow_pad)
        print("Kern size:", kern_size)
        self.imsize = imsize
        if outsize is None:
            self.outsize = imsize
        else:
            self.outsize = outsize
            
        self.layers = nn.Sequential(
            nn.Conv2d(imsize[0], imsize[0], kernel_size=1, padding=pow_pad), #32x32x1 = 1024
            nn.BatchNorm2d(imsize[0]),
            nn.ReLU(),
            nn.Conv2d(imsize[0], 1*s, kernel_size=5, padding=2), #32x32x32 = 32768
            nn.BatchNorm2d(1*s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #16x16x32 = 8192
            nn.Conv2d(1*s, 2*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 8x8x64 = 4096
            nn.Conv2d(2*s, 4*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 4x4x128 = 2048
            nn.Conv2d(4*s, 4*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*s),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 2x2x128 = 512
            KernModule(h=kern_size),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(4*s, 4*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(4*s),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(4*s, 2*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(2*s),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(2*s, 1*s, kernel_size=3, padding=1),
            nn.BatchNorm2d(1*s),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(1*s, self.outsize[0], kernel_size=5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''Forward pass through network'''
        N = x.shape[0]
        out = self.layers(x)[..., :self.outsize[-2], :self.outsize[-1]]
        return out
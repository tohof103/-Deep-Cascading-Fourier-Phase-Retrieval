#Used with permission from Alexander Oberstraß and Tobias Uelwer
import torch

def ifft2(batch_ft):
    '''Returns inverse of 2D-discrete fft of given batch'''
    return torch.irfft(batch_ft, signal_ndim=2, onesided=False)

def fft2(batch):
    '''Returns 2D-discrete fft of given batch'''
    return torch.squeeze(torch.rfft(batch, 2, onesided=False))

def fft2WoSq(batch):
    '''Returns 2D-discrete fft of given batch without squeezing'''
    return torch.rfft(batch, 2, onesided=False)

def magnitude(batch):
    '''Returns magnitude of given batch'''
    return torch.sqrt(torch.sum(batch**2, -1))

def squaredmagnitude(batch):
    '''Returns squared magnitude of given batch'''
    return torch.sum(batch**2, -1)

def real(batch_ri):
    '''Returns real part of given batch'''
    return batch_ri[...,0]

def imag(batch_ri):
    '''Returns imaginary part of given batch'''
    return batch_ri[...,1]

def phase(batch):
    '''Returns phase of given batch'''
    return torch.atan2(imag(batch), real(batch))


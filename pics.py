import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
import torch.nn.functional as F

from fourier import fft2, fft2WoSq, magnitude, phase

#Used with permission from Alexander Oberstraß and Tobias Uelwer    
def plot_grid(images, grid_size=8, figsize=None, save = False):
    """
    Expects 4d tensor with shape (B, C, H, W)
    save: boolean
        Decides wether to save grid
    """
    if type(images) is np.ndarray:
        images = torch.from_numpy(images).float()
    images_concat = torchvision.utils.make_grid(images, nrow=grid_size, padding=2, pad_value=255)
    if save:
        torchvision.utils.save_image(images_concat, 'plots/Reconstructions.png')
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(images_concat.numpy(), (1,2,0)), interpolation='nearest')
    plt.show()
    
def greypic(colors, size): 
    '''Returns batch of grey images with given size'''
    return torch.ones(64, colors, size, size)/100

def blackpic(colors, size): 
    '''Returns batch of black images with given size'''
    return torch.zeros(64, colors, size, size)

def meanpic(train_data):  
    '''Returns batch of images with the mean brightness of the first batch of train_data'''   
    for data in train_data:
        return torch.mean(data, dim = 0) 
    
def DFPRwoPhsaveToPDF(nets, dataloader, startpic, NUM_Of_Nets, device):    
    '''
    Prints targets and reconstructions from DFPRwoPh
    Parameters:
    -----------
        nets: nn.module array
            Networks to test
        dataloader: dataloader
            Dataloader of data to use
        startpic: float array size of images
            Start image to feed to first network
        NUM_Of_Nets: int
            Number of networks to test
        device: torch.device
            Device to test on
    Returns:
    --------
        -   
    '''
    idx = 4
    for net in nets:
        net.eval()
    
    for picbatch in dataloader:
        target = picbatch.to(device)
        break
        
    torchvision.utils.save_image(target[idx], 'plots/Target.png')
    x = startpic.to(device)
    data = magnitude(fft2(target)).to(device)     
    torchvision.utils.save_image(x[idx], 'plots/startpic.png')
    
    for step in range(NUM_Of_Nets):
        delta_y = (magnitude(fft2(x)) - data).to(device)
        out = nets[step](delta_y)
        out = out * ((0.5-(step%2))*2)
        x = torch.clamp(x + out, 0, 1)
        path = 'plots/Reconstruction{}.png'.format(step)
        torchvision.utils.save_image(x[idx], path)
        
    
def DFPRsaveToPDF(nets, dataloader, startpic, NUM_Of_Nets, device):    
    '''
    Prints targets and reconstructions from DFPRwoPh
    Parameters:
    -----------
        nets: nn.module array
            Networks to test
        dataloader: dataloader
            Dataloader of data to use
        startpic: float array size of images
            Start image to feed to first network
        NUM_Of_Nets: int
            Number of networks to test
        device: torch.device
            Device to test on
    Returns:
    --------
        -   
    '''
    idx = 4
    for net in nets:
        net.eval()
    
    for picbatch in dataloader:
        target = picbatch.to(device)
        break
        
    torchvision.utils.save_image(target[idx], 'plots/Target.png')
    x = startpic.to(device)
    measurement = magnitude(fft2WoSq(target)).to(device)
    torchvision.utils.save_image(measurement[idx], 'plots/Magnitude.png')
    phs = phase(fft2WoSq(target))
    torchvision.utils.save_image(phs[idx], 'plots/Phase.png')     
    torchvision.utils.save_image(x[idx], 'plots/startpic.png')
    
    for step in range(NUM_Of_Nets):
        x = x.detach()            
        fourier_t = fft2WoSq(x)            
        phase_x = phase(fourier_t)
        delta_y = magnitude(fourier_t) - measurement
        data = torch.cat([delta_y, phase_x], 1).detach()
        out = nets[step](data)

        out = out * ((0.5-(step%2))*2)    #alternating addition and substraction        
        x = torch.clamp(x + out, 0, 1)
        
        path = 'plots/Reconstruction{}.png'.format(step)
        torchvision.utils.save_image(x[idx], path)
        
    
def downsamplingsaveToPDF(nets, dataloader, NUM_Of_Nets, device, im_size):    
    '''
    Prints targets and reconstructions from DFPRwoPh
    Parameters:
    -----------
        nets: nn.module array
            Networks to test
        dataloader: dataloader
            Dataloader of data to use
        im_size: int array
            Array of reconstruction sizes
        NUM_Of_Nets: int
            Number of networks to test
        device: torch.device
            Device to test on
    Returns:
    --------
        -   
    '''
    idx = 4
    for net in nets:
        net.eval()
        
    for picbatch in dataloader:
        target = picbatch.to(device)
        break
    
    torchvision.utils.save_image(target[idx], 'plots/Target.png')
    ft = torch.from_numpy(np.fft.fftshift(fft2WoSq(target).cpu().numpy()))
    real = ft[...,0][idx]
    torchvision.utils.save_image(real/torch.max(real), 'plots/Real.png')
    img = ft[...,1][idx]
    torchvision.utils.save_image(img/torch.max(img), 'plots/Img.png')
    phs = phase(ft)[idx]
    norm_magn = magnitude(ft)[idx]    
    torchvision.utils.save_image(phs/torch.max(phs), 'plots/Phase.png')
    torchvision.utils.save_image(norm_magn/torch.max(norm_magn), 'plots/Magnitude.png')
    
    for step in range(NUM_Of_Nets):        
        stage_target = F.interpolate(target, size=(im_size[step], im_size[step]))
        path = 'plots/StageTarget{}.png'.format(step)
        torchvision.utils.save_image(stage_target[idx], path)            
        data = magnitude(fft2WoSq(target)).to(device)
            
        if step > 0:
            out = F.interpolate(out, size=(im_size[-1],im_size[-1]))  
            data = torch.cat([data, out], 1)
        out = nets[step](data)
        path = 'plots/Reconstruction{}.png'.format(step)
        torchvision.utils.save_image(out[idx], path)
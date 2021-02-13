#Used with permission from Alexander Oberstraß and Tobias Uelwer
import torch
import numpy as np
from skimage.metrics import structural_similarity  as _ssim
   
def mse(predicted_images, true_images):
    '''
    Expect shape of N x (1,3) x X x Y
    Returns mean-squared error and its standard deviation
    '''
    N = len(true_images)
    total_se = np.empty((N))
    
    for i in range(len(true_images)):    
        total_se[i] = np.mean((true_images[i] - predicted_images[i]) ** 2)
        
    return np.mean(total_se), np.std(total_se)   

def ssim(predicted_images, true_images):
    '''
    Expect shape of N x (1,3) x X x Y
    Returns structural similarity index and its standard deviation
    '''
    assert(predicted_images.shape == true_images.shape)
    assert(predicted_images.shape[-3] <= 3)
    N = len(true_images)
    total_ssim = np.empty((N))

    for i in range(len(true_images)):    
        total_ssim[i] = _ssim(true_images[i].transpose(1, 2, 0),
                              predicted_images[i].transpose(1, 2, 0),
                              multichannel=True)
        
    return np.mean(total_ssim), np.std(total_ssim)

def mae(predicted_images, true_images):
    '''
    Expect shape of N x (1,3) x X x Y
    Returns mean-absolute error and its standard deviation
    '''
    N = len(true_images)
    total_ae = np.empty((N))
    
    for i in range(len(true_images)):    
        total_ae[i] = np.mean(np.abs(true_images[i] - predicted_images[i]))
        
    return np.mean(total_ae), np.std(total_ae)


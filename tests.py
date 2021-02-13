#Except test functions used with permission from Alexander Oberstraß and Tobias Uelwer
from fourier import fft2, fft2WoSq, magnitude, phase
import torch
import numpy as np
import math
from skimage.registration import phase_cross_correlation
from scipy.signal import convolve2d
import torch.nn.functional as F

from pics import plot_grid
from losses import mae, mse, ssim

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def phase_correlation(moving, fixed):
    assert(moving.shape == fixed.shape)

    if moving.shape[-1] == 3:
        moving_gray = rgb2gray(moving)
        fixed_gray = rgb2gray(fixed)
    elif moving.shape[-1] == 1:
        moving_gray = moving[..., 0]
        fixed_gray = fixed[..., 0]
    else:
        print("Image channel Error!")
    
    ft_moving = np.fft.fft2(moving_gray)
    ft_fixed = np.fft.fft2(fixed_gray)
    prod = ft_moving * ft_fixed.conj()
    a = (prod / (np.abs(prod) + 1e-16))
    corr = np.fft.ifft2(a)
    corr_max = np.max(corr)
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    out = np.roll(moving, -1*np.array(idx), axis=(0, 1))
    return out, corr_max

def cross_correlation(moving, fixed):    
    if moving.shape[-1] == 3:
        moving_gray = rgb2gray(moving)
        fixed_gray = rgb2gray(fixed)
    elif moving.shape[-1] == 1:
        moving_gray = moving[..., 0]
        fixed_gray = fixed[..., 0]
    else:
        print("Image channel Error!")
    
    shift, error, diffphase = phase_cross_correlation(moving_gray, fixed_gray)
    out = np.roll(moving, -np.array(shift).astype(np.int), axis=(0, 1))
    return out, error

def register_croco(predicted_images, true_images, torch=True):
    pred_reg = np.empty(predicted_images.shape, dtype=predicted_images.dtype)

    for i in range(len(true_images)):
        if torch:
            true_image = true_images[i].transpose(1, 2, 0)
            predicted_image = predicted_images[i].transpose(1, 2, 0)
        else:
            true_image = true_images[i]
            predicted_image = predicted_images[i]

        shift_predict, shift_error = cross_correlation(predicted_image, true_image)
        rotshift_predict, rotshift_error = cross_correlation(np.rot90(predicted_image, k=2, axes=(0, 1)), true_image)
        
        if torch:
            pred_reg[i] = shift_predict.transpose(2, 0, 1) if shift_error <= rotshift_error else rotshift_predict.transpose(2, 0, 1)
        else:
            pred_reg[i] = shift_predict if shift_error <= rotshift_error else rotshift_predict
        
    return pred_reg

def sharp_dist(predicted_images, true_images):
    if predicted_images.shape[1]==3:
        predicted_images_gray = np.transpose(predicted_images,(0,2,3,1))
        predicted_images_gray = rgb2gray(predicted_images_gray)[:, None]
    else:
        predicted_images_gray = predicted_images
    
    if true_images.shape[1]==3:
        true_images_gray = np.transpose(true_images,(0,2,3,1))
        true_images_gray = rgb2gray(true_images_gray)[:, None]
    else:
        true_images_gray = true_images
    
    dists = []
    for true, predicted in zip(true_images_gray, predicted_images_gray):
        f = np.array([[1,-1]])
        filtered_true = convolve2d(true[0], f, mode='valid')[:-1,:]\
            +convolve2d(true[0], f.T, mode='valid')[:,:-1]
        filtered_predicted = convolve2d(predicted[0], f,mode='valid')[:-1,:]\
            +convolve2d(predicted[0], f.T, mode='valid')[:,:-1]
        dists.append(np.mean(np.abs(filtered_true-filtered_predicted)))
    return np.mean(dists), np.std(dists)

def register_phaco(predicted_images, true_images, torch=True):
    pred_reg = np.empty(predicted_images.shape, dtype=predicted_images.dtype)
    
    for i in range(len(true_images)):
        if torch:
            true_image = true_images[i].transpose(1, 2, 0)
            predicted_image = predicted_images[i].transpose(1, 2, 0)
        else:
            true_image = true_images[i]
            predicted_image = predicted_images[i]
            
        shift_predict, shift_corr = phase_correlation(predicted_image, true_image)
        rotshift_predict, rotshift_corr = phase_correlation(np.rot90(predicted_image, k=2, axes=(0, 1)), true_image)
        
        if torch:
            pred_reg[i] = shift_predict.transpose(2, 0, 1) if shift_corr >= rotshift_corr else rotshift_predict.transpose(2, 0, 1)
        else:
            pred_reg[i] = shift_predict if shift_corr_max >= rotshift_corr_max else rotshift_predict

    return pred_reg

def magn_mse(predicted_images, true_images):
    # Expect shape of N x (1,3) x X x Y
    N = len(true_images)
    total_se = np.empty((N))
    
    for i in range(len(true_images)):    
        true_image = true_images[i]
        predicted_image = predicted_images[i]

        true_magn = np.abs(np.fft.fft2(true_image))
        predicted_magn = np.abs(np.fft.fft2(predicted_image))

        total_se[i] = np.mean((true_magn - predicted_magn) ** 2)
        
    return np.mean(total_se), np.std(total_se)

def benchmark(pred, true, check_all=False, check=["mse", "magn", "imcon"]):
    '''
    Benchmark parcours to evaluate quality of predictions.
    Checks include MSE, MAE and SSIM applied directly, using phase-correlation and cross-correllation.
    Sharpness and the magnitude error are tested
    and image constraints are checked.
    Method expects numpy arrays
    '''    
    pred_signal = np.real(pred)
    true_signal = np.real(true)
    
    checks = [e.lower() for e in check]
    
    pred_croco = register_croco(pred_signal, true_signal)
    pred_phaco = register_phaco(pred_signal, true_signal)
    
    markdown = ""
    
    print("Signal error:")
    if "mse" in checks or check_all:
        _mse = mse(pred_signal, true_signal)
        markdown = markdown + " {:.{}f} |".format(_mse[0], 4 + math.floor(-math.log10(_mse[0])))
        print("  MSE:        {}, std: {}".format(*_mse))
    if "mae" in checks or check_all:
        _mae = mae(pred_signal, true_signal)
        markdown = markdown + " {:.{}f} |".format(_mae[0], 4 + math.floor(-math.log10(_mae[0])))
        print("  MAE:        {}, std: {}".format(*_mae))
    if "ssim" in checks or check_all:
        _ssim = ssim(pred_signal, true_signal)
        markdown = markdown + " {:.{}f} |".format(_ssim[0], 4 + math.floor(-math.log10(_ssim[0])))
        print("  SSIM:       {}, std: {}".format(*_ssim))
    if "sharpness" in checks or check_all:
        _sharpness = sharp_dist(pred_croco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_sharpness[0], 4 + math.floor(-math.log10(_sharpness[0])))
        print("  Sharpness:  {}, std: {}".format(*_sharpness))
    if "phaco" in checks or check_all:
        print("=============================PHACO=============================")
        _fasimse = mse(pred_phaco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_fasimse[0], 4 + math.floor(-math.log10(_fasimse[0])))
        print("  PhaCo-MSE:  {}, std: {}".format(*_fasimse))
        _fasimae = mae(pred_phaco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_fasimae[0], 4 + math.floor(-math.log10(_fasimae[0])))
        print("  PhaCo-MAE:  {}, std: {}".format(*_fasimae))
        _fasissim = ssim(pred_phaco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_fasissim[0], 4 + math.floor(-math.log10(_fasissim[0])))
        print("  PhaCo-SSIM: {}, std: {}".format(*_fasissim))
    if "croco" in checks or check_all:
        print("=============================CROCO=============================")
        _crocomse = mse(pred_croco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_crocomse[0], 4 + math.floor(-math.log10(_crocomse[0])))
        print("  CroCo-MSE:  {}, std: {}".format(*_crocomse))
        _crocomae = mae(pred_croco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_crocomae[0], 4 + math.floor(-math.log10(_crocomae[0])))
        print("  CroCo-MAE:  {}, std: {}".format(*_crocomae))
        _crocossim = ssim(pred_croco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_crocossim[0], 4 + math.floor(-math.log10(_crocossim[0])))
        print("  CroCo-SSIM: {}, std: {}".format(*_crocossim))
    if "magn" in checks or check_all:
        _magn = magn_mse(pred, true)
        markdown = markdown + " {:.{}f} |".format(_magn[0], 4 + math.floor(-math.log10(_magn[0])))
        print()
        print("Magnitude error:")
        print("  MSE Magnitude: {}, std: {}".format(*_magn))
    if "imcon" in checks or check_all:
        print()
        print("Image constraints:")
        print("  Imag part =", np.mean(np.imag(pred)), "- should be very close to 0")
        print("  Real part is in [{0:.2f}, {1:.2f}]".format(np.min(np.real(pred)), np.max(np.real(pred))),
              "- should be in [0, 1]")
    
    print()
    print("Markdown table values:")
    print(markdown)

    
def DFPRtest(nets, test_data, startpic, NUM_Of_Nets, device, pics, save):    
    '''
    Tests and benchmarks performance of networks from DFPR
    Parameters:
    -----------
        nets: nn.module array
            Networks to test
        test_data: dataloader
            Dataloader of test data
        startpic: float array size of images
            Start image to feed to first network
        NUM_Of_Nets: int
            Number of networks to test
        device: torch.device 
            Device to test on
        pics: boolean
            Show grid of pics for visualization
        save: boolean
            Save grid of reconstruction to png
    Returns:
    --------
        -   
    '''
    print("========================================")
    print("Running Tests")
    pred = []
    true = []
    for net in nets:
        net.eval()
    with torch.no_grad():    
        for target in test_data:
            if not len(true):
                true = target
            else:
                true = torch.cat([true, target])                   
            target = target.to(device)
            x = startpic.to(device)
            measurement = magnitude(fft2(target)).to(device)

            for step in range(NUM_Of_Nets):                
                fourier_t = fft2(x)            
                phase_x = phase(fourier_t).detach()
                delta_y = (magnitude(fourier_t) - measurement).to(device)
                delta_y = delta_y.detach()
                data = torch.cat([delta_y, phase_x], 1)
                
                out = nets[step](data)
                out = out * ((0.5-(step%2))*2)
                x = torch.clamp(x + out, 0, 1)
            x = x.cpu()
            if not len(pred):
                pred = x
            else:
                pred = torch.cat([pred, x])
    
    true = true.numpy()
    pred = pred.numpy()    
    
    if pics:
        plot_grid(true[:8], figsize = (15,15), grid_size = 8)
        plot_grid(pred[:8], figsize = (15,15), grid_size = 8, save = save)
           
    benchmark(pred, true, check_all = True)
    print("========================================")

    
def DFPRwoPhtest(nets, test_data, startpic, NUM_Of_Nets, device, pics, save):    
    '''
    Tests and benchmarks performance of networks from DFPR
    Parameters:
    -----------
        nets: nn.module array
            Networks to test
        test_data: dataloader
            Dataloader of test data
        startpic: float array size of images
            Start image to feed to first network
        NUM_Of_Nets: int
            Number of networks to test
        device: torch.device 
            Device to test on
        pics: boolean
            Show grid of pics for visualization
        save: boolean
            Save grid of reconstruction to png
    Returns:
    --------
        -   
    '''
    print("========================================")
    print("Running Tests")
    pred = []
    true = []
    for net in nets:
        net.eval()
    with torch.no_grad():    
        for target in test_data:   
            target = target.to(device)
            if not len(true):
                true = target
            else:
                true = torch.cat([true, target])
            x = startpic.to(device)
            measurement = magnitude(fft2(target)).to(device)

            for step in range(NUM_Of_Nets):
                delta_y = (magnitude(fft2(x)) - measurement).to(device)
                
                out = nets[step](delta_y)
                out = out * ((0.5-(step%2))*2)
                x = torch.clamp(x + out, 0, 1)

            if not len(pred):
                pred = x
            else:
                pred = torch.cat([pred, x])
    
    true = true.cpu().numpy()
    pred = pred.cpu().numpy()    
    
    if pics:
        plot_grid(true[:8], figsize = (15,15), grid_size = 8)
        plot_grid(pred[:8], figsize = (15,15), grid_size = 8, save = save)
           
    benchmark(pred, true, check_all = True)
    print("========================================")
    

def downsampling_test(nets, test_data, NUM_Of_Nets, device, im_size, pics, save):    
    '''
    Tests and benchmarks performance of networks from DFPR
    Parameters:
    -----------
        nets: nn.module array
            Networks to test
        test_data: dataloader
            Dataloader of test data
        NUM_Of_Nets: int
            Number of networks to test
        device: torch.device 
            Device to test on
        im_size: int array
            Array of reconstruction sizes
        pics: boolean
            Show grid of pics for visualization
        save: boolean
            Save grid of reconstruction to png
    Returns:
    --------
        -   
    '''
    print("========================================")
    print("Running Tests")
    pred = []
    true = []
    for net in nets:
        net.eval()
    with torch.no_grad():
        for target in test_data:
            if not len(true):
                true = target
            else:
                true = torch.cat([true, target])
            target = target.to(device)                     
            for step in range(NUM_Of_Nets):
                stage_target = F.interpolate(target, size=(im_size[step], im_size[step]))
                data = magnitude(fft2WoSq(target))            
                if step > 0:
                    out = F.interpolate(out, size=(im_size[-1], im_size[-1]))
                    data = torch.cat([data, out], 1)
                out = nets[step](data)
            out = out.cpu()    
            if not len(pred):
                pred = out
            else:
                pred = torch.cat([pred, out])
    
    true = true.numpy()
    pred = pred.numpy()    
    
    if pics:
        plot_grid(true[:8], figsize = (15,15), grid_size = 8)
        plot_grid(pred[:8], figsize = (15,15), grid_size = 8, save = save)
           
    benchmark(pred, true, check_all = True)
    print("========================================")
    

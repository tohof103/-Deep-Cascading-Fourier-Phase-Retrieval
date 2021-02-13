# Deep-Cascading-Fourier-Phase-Retrieval

## Parameters:
Randomseeds for reproducibility:
```python
torch.manual_seed(17)
random.seed(0)
```

DFPR (100/50 Epochen) for ((x-)MNIST/CelebA):
```python
Net = DFSigmoidNet     #ConvNet for CelebA, DFSigmoidNet for (fashion-)/(E-)MNIST
NUM_Of_Nets = 5
netsize = 1700
lr = 0.0001
startpicgen = greypic
losses = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.L1Loss(), nn.L1Loss()] 
```

DFPR with phase (100/50 Epochen) for ((x-)MNIST/CelebA):
```python
Net = DFSigmoidNet     #ConvNet for CelebA, DFSigmoidNet for (fashion-)/(E-)MNIST
NUM_Of_Nets = 5
netsize = 1700
lr = 0.0001
startpicgen = greypic
losses = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.L1Loss(), nn.L1Loss()] 
```

MNIST-CasPR (100 Epochen):
```python
Net = DCSigmoidNet     
Num_Of_Nets = 5
netsizes = [1100, 1300, 1500, 1800, 1800]
lr = 0.0001
im_size = [7,14,21,28,28]
tau = 0 #for mag-loss: 2e-6
losses = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()]
```
fashion-CasPR (100 Epochen):
```python
Net = DCSigmoidNet     
Num_Of_Nets = 5
netsizes = [1100, 1300, 1500, 1800, 1800]
lr = 0.0001
im_size = [7,14,21,28,28]
tau = 0 #for mag-loss: 1e-7
losses = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.L1Loss()]
```
EMNIST-CasPR (100 Epochen):
```python
Net = DCSigmoidNet     
Num_Of_Nets = 5
netsizes = [1100, 1300, 1500, 1800, 1800]
lr = 0.0001
im_size = [7,14,21,28,28]
tau = 0 #for mag-loss: 1e-6
losses = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.L1Loss()]
```
CelebA-CasPR (50 epochs):

```python
Net = ConvNet     
Num_Of_Nets = 5
netsizes = [1100, 1300, 1500, 1800, 1800]
lr = 0.0001
im_size = [16,32,48,64,64]
tau = 0 #for mag-loss: 1e-10
losses = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.L1Loss()]
```
:warning: **Values for magnitude loss without new Dropout**

## Best values so far:
| Dataset/ loss | MNIST: MSE    | MNIST: MAE    | MNIST: SSIM | fashion: MSE  | fashion: MAE  | fashion: SSIM |
|---------------|---------------|---------------|-------------|---------------|---------------|---------------|
| E2E           | 0.0183        | 0.0411        | 0.8345      | 0.0128        | 0.0526        | 0.7940        |
| DF-PR         | 0.0146        | 0.0400        | 0.8535      | 0.0122        | 0.0538        | 0.7899        |
| DF-PR-phase   | 0.0123        | 0.0337        | 0.8824      | 0.0113        | 0.0505        | 0.8065        |
| Cas-PR / tau  | 0.0123/0.0124 | 0.0374/0.0370 | 0.8754/0.876| 0.0114/0.0120 | 0.0494/0.0512 | 0.8077/0.800  |


| Dataset/ loss | EMNIST: MSE   | EMNIST: MAE   | EMNIST: SSIM | CelebA: MSE     | CelebA: MAE    | CelebA: SSIM   |
|---------------|---------------|---------------|--------------|-----------------|----------------|----------------|
| E2E           | 0.0229        | 0.0657        | 0.7849       | 0.0106          | 0.0699         | 0.7444         |
| DF-PR         | 0.0208        | 0.0612        | 0.8029       | 0.0107          | 0.0723         | 0.7217         |
| DF-PR-phase   | 0.0162        | 0.0494        | 0.8637       | 0.00960         | 0.0678         | 0.7396         |
| Cas-PR / tau  | 0.0165/0.0146 | 0.0473/0.0503 | 0.870/0.868  | 0.00960/0.0101  | 0.0670/0.0704  | 0.7462/0.0727  |

#Used with permission from Alexander Oberstraß and Tobias Uelwer
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler,Sampler
from PIL import Image
import io
import h5py

 
class CelebAH5(torch.utils.data.Dataset):
    '''Custom Dataset for loading the CelebA H5 file containing all images as jpeg'''
    def __init__(self, h5file, transform):
        super(CelebAH5, self).__init__()
        self.h5file = h5file
        self.n_images = self.h5file['images'].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        bin_data =  self.h5file['images'][index]
        return self.transform(Image.open(io.BytesIO(bin_data)))

    def __len__(self):
        return self.n_images
    

class ImageOnly(torch.utils.data.Dataset):
    '''Discards all meta information expect the image (As needed for MNIST)'''
    def __init__(self, orig_dataset):
        self.orig_dataset = orig_dataset

    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        return self.orig_dataset[idx][0]


class LinearSampler(Sampler):
    '''Linearly samples from a dataset without shuffle (Needed for linear validation)'''
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)        

        
def load(name='MNIST', path="", batch_size=64, normalization=[]):
    '''
    Loads dataset and saves it to three dataloader for traing, validation and testing
    Optional normalization to images
    Parameters:
    -----------
        name (optional): string
            Name of dataset to load, implemented are MNIST, Fashion(-MNIST),
            EMNIST and CelebA, default is MNIST
        path (optional): string
            Path to dataset if not downloading from standard dataset library
        batch_size (optional): int
            Defines batchsize to part data into, default is 64
        normaization (optional): Normalization to transform data if wanted
    Returns:
    --------
        array of dataloader: (test_data, validation_data, test_data)
        imsize: size of images
    '''
    if name[:6].lower() == 'celeba':
        print(name)
        
        if name[6:].lower() == 'pad':
            trans = transforms.Compose([
                transforms.CenterCrop((108, 108)),
                transforms.Resize(64),
                transforms.Pad(32, 0),
                transforms.ToTensor(), 
            ]+normalization)
        else: 
            trans = transforms.Compose([
            transforms.CenterCrop((108, 108)),
            transforms.Resize(64),
            transforms.ToTensor(),
        ]+normalization)

        h5file = h5py.File(path, 'r')

        dataset = CelebAH5(h5file, transform=trans)

        indices = list(range(202589))
        train_idx, valid_idx, test_idx = indices[:162769], indices[162769:182636], indices[182636:183660]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = LinearSampler(valid_idx)
        test_sampler = LinearSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
        )

        validloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=valid_sampler, shuffle=False, drop_last=True
        )

        testloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False, drop_last=True
        )

        imsize = (3, 64, 64)
            
        return {'train': trainloader, 'val': validloader, 'test': testloader}, imsize
    
    if name[:5].lower() == 'mnist':
        print(name)
        
        
        train_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        val_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        test_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        
        trainset = datasets.MNIST(
            root=path, train=True,
            download=True,
            transform=train_trans
        )
        valset = datasets.MNIST(
            root=path, train=False,
            download=True,
            transform=val_trans
        )                                      
        testset = datasets.MNIST(
            root=path, train=False,
            download=True,
            transform=test_trans
        )
        
        trainset = ImageOnly(trainset)
        valset = ImageOnly(valset)
        testset = ImageOnly(testset)
        
        indices = list(range(10000))
        valid_idx, test_idx = indices[-2048:], indices[:-8976] # Last 2048 val, First 1024 test
        valid_sampler = LinearSampler(valid_idx)
        test_sampler = LinearSampler(test_idx)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, drop_last=True
        )
            
        validloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, sampler=valid_sampler, shuffle=False, drop_last=True
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=test_sampler, shuffle=False, drop_last=True
        )

        imsize = (1, 28, 28)
        
        return {'train': trainloader, 'val': validloader, 'test': testloader}, imsize

    if name.lower() == 'fashion':
        print(name)
        
        train_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        val_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        test_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        
        trainset = datasets.FashionMNIST(
            root=path, train=True,
            download=True,
            transform=train_trans
        )
        valset = datasets.FashionMNIST(
            root=path, train=False,
            download=True,
            transform=val_trans
        )                                      
        testset = datasets.FashionMNIST(
            root=path, train=False,
            download=True,
            transform=test_trans
        )
        
        trainset = ImageOnly(trainset)
        valset = ImageOnly(valset)
        testset = ImageOnly(testset)
        
        indices = list(range(10000))
        valid_idx, test_idx = indices[-2048:], indices[:-8976] # Last 2048 val, First 1024 test
        valid_sampler = LinearSampler(valid_idx)
        test_sampler = LinearSampler(test_idx)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, drop_last=True
        )
            
        validloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, sampler=valid_sampler, shuffle=False, drop_last=True
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=test_sampler, shuffle=False, drop_last=True
        )

        imsize = (1, 28, 28)
        
        return {'train': trainloader, 'val': validloader, 'test': testloader}, imsize
    
    if name.lower() == 'emnist':
        print(name)
        
        train_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        val_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        test_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        
        trainset = datasets.EMNIST(
            root=path, split='balanced', train=True,
            download=True,
            transform=train_trans
        )
        valset = datasets.EMNIST(
            root=path, split='balanced', train=False,
            download=True,
            transform=val_trans
        )                                      
        testset = datasets.EMNIST(
            root=path, split='balanced', train=False,
            download=True,
            transform=test_trans
        )
        
        trainset = ImageOnly(trainset)
        valset = ImageOnly(valset)
        testset = ImageOnly(testset)
        
        indices = list(range(10000))
        valid_idx, test_idx = indices[-2048:], indices[:-8976] # Last 2048 val, First 1024 test
        valid_sampler = LinearSampler(valid_idx)
        test_sampler = LinearSampler(test_idx)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, drop_last=True
        )
            
        validloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, sampler=valid_sampler, shuffle=False, drop_last=True
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=test_sampler, shuffle=False, drop_last=True
        )

        imsize = (1, 28, 28)
        
        return {'train': trainloader, 'val': validloader, 'test': testloader}, imsize
   
    print("{} did not match any known dataset".format(name))
    return None

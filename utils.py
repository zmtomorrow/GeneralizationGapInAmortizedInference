import torchvision
from torch.utils import data
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt


rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5


def gray_show_many(image,number_sqrt):
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
    plt.axis('off')
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()


def tensor_to_ndarray(tensor):
    if type(tensor) is tuple:
        return tuple(tensor_to_ndarray(t) for t in tensor)
    else:
        return tensor.detach().numpy()

def ndarray_to_tensor(arr):
    if type(arr) is tuple:
        return tuple(ndarray_to_tensor(a) for a in arr)
    elif type(arr) is torch.Tensor:
        return arr
    else:
        return torch.from_numpy(np.float32(arr))

def torch_fun_to_numpy_fun(fun):
    def numpy_fun(*args, **kwargs):
        torch_args = ndarray_to_tensor(args)
        return tensor_to_ndarray(fun(*torch_args, **kwargs))
    return numpy_fun

def LoadData(opt):
    if opt['data_set'] == 'SVHN':
        train_data=torchvision.datasets.SVHN(opt['dataset_path'], split='train', download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.SVHN(opt['dataset_path'], split='test', download=False,transform=torchvision.transforms.ToTensor())
        
    elif opt['data_set'] == 'CIFAR':
        if opt['data_aug']==True:
            transform=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor()])
        else:
            transform=torchvision.transforms.ToTensor()
        train_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=True, download=False,transform=transform)
        test_data=torchvision.datasets.CIFAR10(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())

    elif opt['data_set']=='MNIST':
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=torchvision.transforms.ToTensor())
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=torchvision.transforms.ToTensor())
    
    elif opt['data_set']=='BinaryMNIST':
        trans=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: torch.round(x),
        ])
        train_data=torchvision.datasets.MNIST(opt['dataset_path'], train=True, download=False,transform=trans)
        test_data=torchvision.datasets.MNIST(opt['dataset_path'], train=False, download=False,transform=trans)
    
    else:
        raise NotImplementedError

    train_data_loader = data.DataLoader(train_data, batch_size=opt['batch_size'], shuffle=True)
    test_data_loader = data.DataLoader(test_data, batch_size=opt['test_batch_size'], shuffle=False)
    train_data_evaluation = data.DataLoader(train_data, batch_size=opt['test_batch_size'], shuffle=False)
    return train_data_loader,test_data_loader,train_data_evaluation

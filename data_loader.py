import torch
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.types import Device

from pyutils.general import logger

class MnistData():
    def __init__(self, 
                 mode: str = 'spatial',
                 normalize: bool = True,
                 batchSize: int = 4,
                 numComponents: int = None
                 ):
        '''
        Loads MNIST data from torchvision datasets
        Inputs -
            mode        str, load data in spatial or fourier format
            normalize   bool, True if data is to be normalized
            batchSize   int, batch size
        Outputs -
            trainLoader DataLoader, training data loader
            testLoader  DataLoader, test data loader
        '''
        assert mode in {'spatial', 'fourier', 'digital'}, logger.error(
            f"Mode not supported. Expected one from (spatial, fourier, digital) but got {mode}."
        )

        self.batchSize = batchSize
        self.numComponents = numComponents

        if mode in {'spatial', 'digital'}:
            self.spatialTransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if normalize:
                self.spatialTransform.transforms.insert(
                    1,transforms.Normalize((0.1307,), (0.3081,))
                )
            self.targetTransform = transforms.Lambda(
                lambda y: torch.zeros(10,dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            )
            self.trainData = MNIST(
                root="data",
                train=True,
                download=True,
                transform=self.spatialTransform,
                target_transform=self.targetTransform
            )
            self.testData = MNIST(
                root="data",
                train=False,
                download=True,
                transform=self.spatialTransform,
                target_transform=self.targetTransform
            )
            self.trainLoader = DataLoader(self.trainData, batch_size=self.batchSize,
                                          shuffle=True, num_workers=4)
            self.testLoader = DataLoader(self.testData, batch_size=self.batchSize,
                                         shuffle=False, num_workers=4)
        elif mode == 'fourier':
            self.fourierTransform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: torch.fft.fftshift(torch.fft.fft2(x), dim=(-2,-1)))
            ])
            if numComponents is not None:
                 # we know that mnist has images of size 28x28
                 idx1 = (28 - self.numComponents)//2
                 idx2 = idx1 + self.numComponents
                 self.fourierTransform.transforms.insert(
                      3, transforms.Lambda(lambda x: x[:, idx1:idx2, idx1:idx2])
                 )
            if normalize:
                self.fourierTransform.transforms.insert(
                    #4, transforms.Lambda(self.__complex_norm_polar)
                    4, transforms.Lambda(lambda x: torch.complex((x.real - -0.8315)/32.7469, (x.imag - 2.116e-5)/20.5933))
                )
            self.targetTransform = transforms.Lambda(
                lambda y: torch.zeros(10,dtype=torch.double).scatter_(0, torch.tensor(y), value=1)
            )
            self.trainData = MNIST(
                root="data",
                train=True,
                download=True,
                transform=self.fourierTransform,
                target_transform=self.targetTransform
            )
            self.testData = MNIST(
                root="data",
                train=False,
                download=True,
                transform=self.fourierTransform,
                target_transform=self.targetTransform
            )
            self.trainLoader = DataLoader(self.trainData, batch_size=self.batchSize,
                                          shuffle=True, num_workers=4)
            self.testLoader = DataLoader(self.testData, batch_size=self.batchSize,
                                         shuffle=False, num_workers=4)
        else:
             raise NotImplementedError("Only spatial, digital and fourier modes supported")

    def __complex_norm_polar(self, x):
        upper = 1.0
        rad = torch.abs(x)
        max_rad = torch.max(rad)
        norm_rad = rad*upper/max_rad
        x_angle = x.angle()
        return torch.view_as_complex(torch.stack([norm_rad*torch.cos(x_angle), norm_rad*torch.sin(x_angle)], dim=-1))
    
    def getDataLoaders(self):
            return self.trainLoader, self.testLoader
    
class CifarData():
    def __init__(self, 
                 mode: str = 'spatial',
                 normalize: bool = True,
                 batchSize: int = 4):
        '''
        Loads Cifar data from torchvision datasets
        Inputs -
            mode        str, load data in spatial or fourier format
            normalize   bool, True if data is to be normalized
            batchSize   int, batch size
        Outputs -
            trainLoader DataLoader, training data loader
            testLoader  DataLoader, test data loader
        '''
        assert mode in {'spatial', 'fourier', 'digital'}, logger.error(
            f"Mode not supported. Expected one from (spatial, fourier, digital) but got {mode}."
        )

        self.batchSize = batchSize

        if mode in {'spatial', 'digital'}:
            self.spatialTransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if normalize:
                self.spatialTransform.transforms.insert(
                    1,transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                )
            self.targetTransform = transforms.Lambda(
                lambda y: torch.zeros(10,dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            )
            self.trainData = CIFAR10(
                root="data",
                train=True,
                download=True,
                transform=self.spatialTransform,
                target_transform=self.targetTransform
            )
            self.testData = CIFAR10(
                root="data",
                train=False,
                download=True,
                transform=self.spatialTransform,
                target_transform=self.targetTransform
            )
            self.trainLoader = DataLoader(self.trainData, batch_size=self.batchSize,
                                          shuffle=True, num_workers=4)
            self.testLoader = DataLoader(self.testData, batch_size=self.batchSize,
                                         shuffle=False, num_workers=4)
        elif mode == 'fourier':
            self.fourierTransform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: x.double()),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                #transforms.Grayscale(),
                transforms.Lambda(lambda x: torch.fft.fftshift(torch.fft.fft2(x), dim=(1,2)))
            ])
            if normalize:
                self.fourierTransform.transforms.insert(
                    3, transforms.Lambda(self.__complex_norm_polar)
                    #2, transforms.Lambda(lambda x: x/torch.sqrt(torch.tensor(x.shape[1]*x.shape[2])))
                )
            self.targetTransform = transforms.Lambda(
                lambda y: torch.zeros(10,dtype=torch.float64).scatter_(0, torch.tensor(y), value=1)
            )
            self.trainData = CIFAR10(
                root="data",
                train=True,
                download=True,
                transform=self.fourierTransform,
                target_transform=self.targetTransform
            )
            self.testData = CIFAR10(
                root="data",
                train=False,
                download=True,
                transform=self.fourierTransform,
                target_transform=self.targetTransform
            )
            self.trainLoader = DataLoader(self.trainData, batch_size=self.batchSize,
                                          shuffle=True, num_workers=4)
            self.testLoader = DataLoader(self.testData, batch_size=self.batchSize,
                                         shuffle=False, num_workers=4)
        else:
             raise NotImplementedError("Only spatial, digital and fourier modes supported")

    def __complex_norm_polar(self, x):
        upper = 1.0
        rad = torch.abs(x)
        max_rad = torch.max(rad)
        norm_rad = rad*upper/max_rad
        x_angle = x.angle()
        return torch.view_as_complex(torch.stack([norm_rad*torch.cos(x_angle), norm_rad*torch.sin(x_angle)], dim=-1))
    
    def getDataLoaders(self):
            return self.trainLoader, self.testLoader

class FashionMnistData():
    def __init__(self, 
                 mode: str = 'spatial',
                 normalize: bool = True,
                 batchSize: int = 4,
                 numComponents: int = None):
        '''
        Loads MNIST data from torchvision datasets
        Inputs -
            mode        str, load data in spatial or fourier format
            normalize   bool, True if data is to be normalized
            batchSize   int, batch size
        Outputs -
            trainLoader DataLoader, training data loader
            testLoader  DataLoader, test data loader
        '''
        assert mode in {'spatial', 'fourier', 'digital'}, logger.error(
            f"Mode not supported. Expected one from (spatial, fourier, digital) but got {mode}."
        )

        self.batchSize = batchSize
        self.numComponents = numComponents

        if mode in {'spatial', 'digital'}:
            self.spatialTransform = transforms.Compose([
                transforms.ToTensor(),
            ])
            if normalize:
                self.spatialTransform.transforms.insert(
                    1,transforms.Normalize((0.2860), (0.3526))
                )
            self.targetTransform = transforms.Lambda(
                lambda y: torch.zeros(10,dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            )
            self.trainData = FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=self.spatialTransform,
                target_transform=self.targetTransform
            )
            self.testData = FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=self.spatialTransform,
                target_transform=self.targetTransform
            )
            self.trainLoader = DataLoader(self.trainData, batch_size=self.batchSize,
                                          shuffle=True, num_workers=4)
            self.testLoader = DataLoader(self.testData, batch_size=self.batchSize,
                                         shuffle=False, num_workers=4)
        elif mode == 'fourier':
            self.fourierTransform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860), (0.3526)),
                transforms.Lambda(lambda x: torch.fft.fftshift(torch.fft.fft2(x), dim=(-2,-1)))
            ])
            if numComponents is not None:
                 # we know that fmnist has images of size 28x28
                 idx1 = (28 - self.numComponents)//2
                 idx2 = idx1 + self.numComponents
                 self.fourierTransform.transforms.insert(
                      3, transforms.Lambda(lambda x: x[:, idx1:idx2, idx1:idx2])
                 )
            if normalize:
                self.fourierTransform.transforms.insert(
                    3, transforms.Lambda(self.__complex_norm_polar)
                )
            self.targetTransform = transforms.Lambda(
                lambda y: torch.zeros(10,dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            )
            self.trainData = FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=self.fourierTransform,
                target_transform=self.targetTransform
            )
            self.testData = FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=self.fourierTransform,
                target_transform=self.targetTransform
            )
            self.trainLoader = DataLoader(self.trainData, batch_size=self.batchSize,
                                          shuffle=True, num_workers=4)
            self.testLoader = DataLoader(self.testData, batch_size=self.batchSize,
                                         shuffle=False, num_workers=4)
        else:
             raise NotImplementedError("Only spatial, digital and fourier modes supported")

    def __complex_norm_polar(self, x):
        upper = 1.0
        rad = torch.abs(x)
        max_rad = torch.max(rad)
        norm_rad = rad*upper/max_rad
        x_angle = x.angle()
        return torch.view_as_complex(torch.stack([norm_rad*torch.cos(x_angle), norm_rad*torch.sin(x_angle)], dim=-1))
    
    def getDataLoaders(self):
            return self.trainLoader, self.testLoader
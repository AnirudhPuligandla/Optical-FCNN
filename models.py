import torch
import torch.nn as nn
import torchonn as onn
from torchonn.models import ONNBaseModel
import torch.nn.functional as F

import numpy as np

class simpleCNN(ONNBaseModel):
    def __init__(self,
                 device=torch.device("cuda"),
                 imChannels: int = 1,
                 imSize: int = 28):
        super().__init__()
        '''
        Create a simple CNN model to test on MNIST and CIFAR10 datasets
        Inputs
            device:     device, device to create the model on
            imChannels: int, number of channels in input images
            imSize:     int, size of the images (images assumed to be square) e.g., 28, 32
        '''

        self.conv1 = onn.layers.MZIConv2d(
            in_channels=imChannels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            device=device
        )
        self.conv2 = onn.layers.MZIConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            device=device
        )
        self.pool = nn.MaxPool2d(2)
        # Each conv layer reduce the image size by 2 and pool layers cuts it in half
        denseLayerDims = (imSize - 4)//2
        self.linear1 = onn.layers.MZIBlockLinear(
            in_features=64*denseLayerDims*denseLayerDims,
            out_features=128,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            dtype=torch.float,
            device=device,
        )
        self.linear2 = onn.layers.MZIBlockLinear(
            in_features=128,
            out_features=10,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            dtype=torch.float,
            device=device,
        )
        # Reset parameters
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x= torch.relu(self.conv1(x))
        x=torch.relu(self.conv2(x))
        x=self.pool(x)
        x = x.flatten(1)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
class simpleFCNN(ONNBaseModel):
    def __init__(self,
                 device=torch.device("cuda"),
                 imChannels: int = 1,
                 imSize: int = 28
                 ):
        super().__init__()

        self.g_taper = 1.0
        self.g = 1.75 * np.pi
        self.phi_b = np.pi

        self.imSize = imSize

        # for sptial CNNs, image dimension is reduced by 2 after each convolution layer
        # Set pool size to the same size as in spatial CNNs
        self.poolSize = self.imSize-2
        self.conv1 = onn.layers.FourierConv2d(
            in_channels=imChannels,
            out_channels=32,
            kernel_size=3,
            pool_size=self.poolSize,
            bias=True,
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.poolSize -= 2
        self.denseLayerDims = self.poolSize//2
        self.conv2 = onn.layers.FourierConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            pool_size=self.denseLayerDims,
            bias=True,
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.linear1 = onn.layers.MZIBlockLinear(
            in_features=64*self.denseLayerDims*self.denseLayerDims,
            out_features=128,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            dtype=torch.cfloat,
            photodetect=False,
            device=device,
        )
        self.linear2 = onn.layers.MZIBlockLinear(
            in_features=128,
            out_features=10,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            photodetect=True,
            dtype=torch.cfloat,
            device=device,
        )
        self.EOActivation = onn.layers.ElectroOptic(
            in_features = 25,   # needed only when using bias
            bias = False,
            alpha = 0.1,
            g = self.g * (self.g_taper ** 0), # here, 0 -> i, which in neuroptica usage increments with layers
            phi_b = self.phi_b,
            device=device
        )
        # Reset parameters
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.EOActivation(self.conv1(x))
        x = self.EOActivation(self.conv2(x))
        # clip x to a size similar to maxPooling in spatial CNNs (mnist=12, cifar=14)
        #startIdx = (self.poolSize-self.denseLayerDims)//2
        #endIdx = startIdx + self.denseLayerDims
        #x = x[:,:,startIdx:endIdx,startIdx:endIdx]
        x = x.flatten(1)
        x = self.EOActivation(self.linear1(x))
        x = self.linear2(x)
        x = torch.square(x.real) + torch.square(x.imag)
        #x = self.linear2(x)
        #x = torch.square(x.real) + torch.square(x.imag)
        return x
    
class simpleDCNN(nn.Module):
    def __init__(self,
                 device=torch.device("cuda"),
                 imChannels: int = 1,
                 imSize: int = 28):
        super(simpleDCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=imChannels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            device=device
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            device=device
        )
        self.pool = nn.MaxPool2d(2)
        denseLayerDims = (imSize - 4)//2
        self.linear1 = nn.Linear(
            in_features=64*denseLayerDims*denseLayerDims,
            out_features=128,
            bias=True,
            device=device,
        )
        self.linear2 = nn.Linear(
            in_features=128,
            out_features=10,
            bias=True,
            device=device,
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.linear1(x))
        x = torch.square(torch.abs(self.linear2(x)))
        return x
    
class dummyTest(nn.Module):
    def __init__(self,
                 device=torch.device("cuda"),
                 imChannels: int = 1,
                 imSize: int = 28
                 ):
        super().__init__()

        self.g_taper = 1.0
        self.g = 1.75 * np.pi
        self.phi_b = np.pi

        self.imSize = imSize

        self.convS = onn.layers.MZIConv2d(
            in_channels=imChannels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            mode="usv",
            decompose_alg="clements",
            photodetect=False,
            device=device
        )
        self.poolSize = self.imSize-2
        self.convF = onn.layers.FourierConv2d(
            in_channels=imChannels,
            out_channels=1,
            kernel_size=3,
            pool_size=self.poolSize,
            bias=True,
            mode="weight",
            dtype=torch.cfloat,
            photodetect=False,
            device=device
        )
        self.convL = onn.layers.MZIBlockLinear(
            in_features=self.imSize*self.imSize,
            out_features=26*26,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            dtype=torch.float,
            photodetect=False,
            device=device,
        )
        self.EOActivation = onn.layers.ElectroOptic(
            in_features = 25,   # needed only when using bias
            bias = False,
            alpha = 0.1,
            g = self.g * (self.g_taper ** 0), # here, 0 -> i, which in neuroptica usage increments with layers
            phi_b = self.phi_b,
            device=device
        )
        self.linear = onn.layers.MZIBlockLinear(
            in_features=26*26,
            out_features=10,
            bias=True,
            miniblock=4,
            mode="usv",
            decompose_alg="clements",
            dtype=torch.float,
            photodetect=False,
            device=device,
        )
    def forward(self, x):
        # forier
        #x = self.EOActivation(self.convF(x))
        # spatial
        #x = F.relu(self.convS(x))
        #linear
        x = x.flatten(1)
        x = F.relu(self.convL(x))
        # for all
        x = x.flatten(1)
        x = self.linear(x)
        return x
# %% [code]
import torch

import torch.nn as nn
import math

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(2)
        )
        
    def forward(self,X):
        out = self.layer(X)
        return out
        
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(growth_rate)
        )
    
    def forward(self,X):
        out = self.layer(X)
        out = torch.cat((X, out), 1)
        return out

    
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, expansion=4, dropout_rate=0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate*expansion, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(growth_rate*expansion),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(growth_rate*expansion, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(growth_rate),
        )
    
    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)
        out = torch.cat((X, out), 1)
        return out
    
    

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n=1, block=BasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            block(in_channels , growth_rate, *args, **kwargs),
            *[block(growth_rate*(i+1) + in_channels, 
                    growth_rate, *args, **kwargs) for i in range(n - 1)]
        ])
        
    
    def forward(self, X):
        out = self.blocks[0](X)
        for block in self.blocks[1:]:
            out = block(out)
        return out
    

class DenseNet(nn.Module):
    def __init__(self, in_channels, n_classes, deepths=[16,16,16],growth_rates=[24,24,24], reduction=0.5,*args, **kwargs):
        super().__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels, 2*growth_rates[0], kernel_size=3, padding=1,
                               bias=False)
        
        nb_channels = 2*growth_rates[0]
        
        self.blocks = []
        
        for i, depth in enumerate(deepths):
            self.blocks.append(DenseBlock(nb_channels,growth_rates[i], n=depth,*args, **kwargs))
            nb_channels += growth_rates[i]*depth
            if i != len(deepths)-1:
                self.blocks.append(TransitionBlock(nb_channels, int(math.floor((nb_channels)*reduction))))
                nb_channels = int(math.floor((nb_channels)*reduction))
         
        self.blocks = nn.ModuleList(self.blocks)
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(nb_channels, n_classes),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        out = self.conv1(X)
        for block in self.blocks:
            out = block(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    

class DenseNetProb(nn.Module):
    def __init__(self, in_channels, n_classes, deepths=[16,16,16],growth_rates=[24,24,24], reduction=0.5,*args, **kwargs):
        super().__init__()
        
        
        self.conv1 = nn.Conv2d(in_channels, 2*growth_rates[0], kernel_size=3, padding=1,
                               bias=False)
        
        nb_channels = 2*growth_rates[0]
        
        self.blocks = []
        
        for i, depth in enumerate(deepths):
            self.blocks.append(DenseBlock(nb_channels,growth_rates[i], n=depth,*args, **kwargs))
            nb_channels += growth_rates[i]*depth
            if i != len(deepths)-1:
                self.blocks.append(TransitionBlock(nb_channels, int(math.floor((nb_channels)*reduction))))
                nb_channels = int(math.floor((nb_channels)*reduction))
         
        self.blocks = nn.ModuleList(self.blocks)
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(nb_channels, n_classes),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        out = self.conv1(X)
        for block in self.blocks:
            out = block(out)
        outblocks = self.avg(out)
        outblocks = outblocks.view(outblocks.size(0), -1)
        out = self.fc(outblocks)
        return out, outblocks
        
        
        
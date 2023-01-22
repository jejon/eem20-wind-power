# %% [code]
import torch.nn as nn
import math


def conv_block(input_size, output_size, kernel=(3,3),pool_size=3):
        block = nn.Sequential(
            nn.Conv2d(input_size, output_size,kernel),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride=pool_size-1)
        )

        return block
    
def conv_block3(input_size, output_size, kernel=(3,3),pool_size=3):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size,kernel),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size,kernel),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size,kernel),
        nn.ReLU(),
        nn.MaxPool2d(pool_size, stride=pool_size-1)
    )

    return block

# [(W-K+2P)/S]+1
# W: input volume (width)
# K: kernel size 
# P: padding
# S: Stride

class AlexNet(nn.Module):
    
    
    def __init__(self, nb_input_channels, input_width=64, dropout_rate=0):
        super().__init__()
        
        self.conv1 = conv_block(nb_input_channels, 16, kernel=(5,5))
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block3(32, 64)
        self.flat = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        size_conv1 = math.floor(((input_width-3+2*0)/1 + 1)/2 + (1- 3/2))
        size_conv2 = math.floor(((size_conv1-3+2*0)/1 + 1)/2 + (1- 3/2))
        size_conv3 = math.floor((size_conv2-3+2*0)/1 + 1)
        size_conv4 = math.floor((size_conv3-3+2*0)/1 + 1)
        size_conv5 = math.floor(((size_conv4-3+2*0)/1 + 1)/2 + (1- 3/2))
        
        self.ln = nn.Sequential(
            nn.Linear(int((size_conv5**2)*64), int((size_conv5**2)*64)),
            nn.ReLU(),
            nn.Linear(int((size_conv5**2)*64),int((size_conv5**2)*64)),
            nn.ReLU(),
            nn.Linear(int((size_conv5**2)*64), 1),
            nn.ReLU()
        )
    
    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.dropout1(x)
        out = self.ln(x)
        
        return out
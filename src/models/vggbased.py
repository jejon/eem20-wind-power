import torch.nn as nn


def conv_block2(input_size, output_size, kernel=(3,3),pool_size=(2,2)):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.MaxPool2d(pool_size)
    )

    return block
    
def conv_block3(input_size, output_size, kernel=(3,3),pool_size=(2,2)):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.MaxPool2d(pool_size)
    )

    return block

def conv_block2NoPooling(input_size, output_size, kernel=(3,3),pool_size=(2,2)):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )

    return block
    
def conv_block3NoPooling(input_size, output_size, kernel=(3,3),pool_size=(2,2)):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU(),
        nn.Conv2d(output_size, output_size,kernel, bias=False),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )

    return block

# [(W-K+2P)/S]+1
# W: input volume (width)
# K: kernel size 
# P: padding
# S: Stride

class VGG(nn.Module):
    
    
    def __init__(self, nb_input_channels, input_width=64, dropout_rate=0):
        super().__init__()
        
        self.conv1 = conv_block2(nb_input_channels, 16)
        self.conv2 = conv_block2(16, 32)
        self.conv3 = conv_block3(32, 64)
        self.flat = nn.Flatten()
        self.dropout1 = nn.Dropout(dropout_rate)
        size_conv1 = (input_width-3+2*0)//1 + 1
        size_conv2 = ((size_conv1-3+2*0)//1 + 1)//2
        size_conv3 = (size_conv2-3+2*0)//1 + 1
        size_conv4 = ((size_conv3-3+2*0)//1 + 1)//2
        size_conv5 = (size_conv4-3+2*0)//1 + 1
        size_conv6 = (size_conv5-3+2*0)//1 + 1
        size_conv7 = ((size_conv6-3+2*0)//1 + 1)//2
        
        self.ln = nn.Sequential(
            nn.Linear(int((size_conv7**2)*64), int((size_conv7**2)*64)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int((size_conv7**2)*64),int((size_conv7**2)*64)),
            nn.ReLU(),
            nn.Linear(int((size_conv7**2)*64), 1),
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
    
class VGGLessPooling(nn.Module):
    def __init__(self, nb_input_channels, input_width=64, dropout_rate=0):
        super().__init__()
        
        self.conv1 = conv_block2(nb_input_channels, 64)
        self.conv2 = conv_block2(64, 128)
        self.conv3 = conv_block3NoPooling(128, 128)
        self.conv4 = conv_block3NoPooling(128, 128)
        self.flat = nn.Flatten()
#         self.dropout1 = nn.Dropout(dropout_rate)
        size_conv1 = (input_width-3+2*0)//1 + 1
        size_conv2 = ((size_conv1-3+2*0)//1 + 1)//2
        size_conv3 = (size_conv2-3+2*0)//1 + 1
        size_conv4 = ((size_conv3-3+2*0)//1 + 1)//2
        size_conv5 = (size_conv4-3+2*0)//1 + 1
        size_conv6 = (size_conv5-3+2*0)//1 + 1
        size_conv7 = (size_conv6-3+2*0)//1 + 1
        size_conv8 = (size_conv7-3+2*0)//1 + 1
        size_conv9 = (size_conv8-3+2*0)//1 + 1
        size_conv10 = (size_conv9-3+2*0)//1 + 1
        
        self.ln = nn.Sequential(
            nn.Linear(int((size_conv10**2)*128), 20),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(20,20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        x = self.conv1(X)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flat(x)
#         x = self.dropout1(x)
        out = self.ln(x)
        
        return out
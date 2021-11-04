import torch.nn as nn

def pooling(kernel_size=2, type_pool='max'):
    if type_pool == 'avg':
        return nn.AvgPool2d(kernel_size)
    elif type_pool == 'max':
        return nn.MaxPool2d(kernel_size)
    
class up(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, scale=2, bilinear=True):
        super(up, self).__init__()
        self.conv = nn.Sequential()
        if out_channels is None:
            out_channels = in_channels
        if bilinear or in_channels is None: # bilinear up-sampling
            if out_channels != in_channels:
                self.conv = conv_bn_relu(in_channels,out_channels,1,0)
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear',align_corners=True)
        else:      # deconvolution block
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale, stride=scale)
    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x
        
def conv_bn_relu(in_channels,out_channels,kernel_size=3,padding=1,dilation=1,groups=1):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv

class ResBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResBasicBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.conv(x)
        shortcut = self.shortcut(x)
        return self.relu(residual + shortcut)
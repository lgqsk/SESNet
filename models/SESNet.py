import torch.nn as nn
import torch
from models.parts import pooling, up, conv_bn_relu, ResBasicBlock
from models.parts import ResBasicBlock as BaseModule

class Encoder(nn.Module):
    # The two-stream encoder
    def __init__(self, channels, image_channels=3):
        super(Encoder, self).__init__()
        self.E1 = nn.Sequential(
            conv_bn_relu(image_channels,channels[0],3,1),
            BaseModule(channels[0],channels[0])
        )
        self.E2 = nn.Sequential(
            nn.MaxPool2d(2),
            BaseModule(channels[0],channels[1])
        )
        self.E3 = nn.Sequential(
            nn.MaxPool2d(2),
            BaseModule(channels[1],channels[2])
        )
        self.E4 = nn.Sequential(
            nn.MaxPool2d(2),
            BaseModule(channels[2],channels[3])
        )
        
    def forward(self, I1, I2):

        # Feature Extraction
        e1_1 = self.E1(I1)
        e1_2 = self.E2(e1_1)
        e1_3 = self.E3(e1_2)
        e1_4 = self.E4(e1_3)
        
        e2_1 = self.E1(I2)
        e2_2 = self.E2(e2_1)
        e2_3 = self.E3(e2_2)
        e2_4 = self.E4(e2_3)
        
        # Feature maps of the same scale are concatenated
        e1 = torch.cat([e1_1, e2_1],1)
        e2 = torch.cat([e1_2, e2_2],1)
        e3 = torch.cat([e1_3, e2_3],1)
        e4 = torch.cat([e1_4, e2_4],1)
        
        return e1, e2, e3, e4
    
class get_featureMap_r(nn.Module):
    # Generate the feature map r
    def __init__(self, channels):
        super(get_featureMap_r, self).__init__()
        self.R = nn.Sequential(
            nn.MaxPool2d(2),
            BaseModule(channels[4],channels[4])
        )

    def forward(self, e4):
        r = self.R(e4)
        return r

class SEM(nn.Module):
    # A semantic enhancement module
    def __init__(self, channels, e_channel, scale):
        super(SEM, self).__init__()

        '''
        e_channel: Number of channels of the input feature map ei
        scale: The scale of upsampling of feature map r
        '''

        self.xi = nn.Sequential(
            nn.Conv2d(e_channel, e_channel//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(e_channel//2),
            nn.ReLU(inplace=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(channels[4], e_channel//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(e_channel//2),
            up(scale=scale),
        )

    def forward(self, ei, r):

        # Re-encoding ei
        ei = self.xi(ei)

        # Convert r to weights
        r = self.psi(r)

        # Re-weighting ei
        si = ei * r

        return si

class all_SEM(nn.Module):
    # Combination of all semantic enhancement modules
    def __init__(self, channels):
        super(all_SEM, self).__init__()
        
        self.SEM1 = SEM(channels, channels[1], 16)
        self.SEM2 = SEM(channels, channels[2], 8)
        self.SEM3 = SEM(channels, channels[3], 4)
        self.SEM4 = SEM(channels, channels[4], 2)

    def forward(self, e1, e2, e3, e4, r):

        # r is delivered to each SEM individually
        s1 = self.SEM1(e1, r)
        s2 = self.SEM2(e2, r)
        s3 = self.SEM3(e3, r)
        s4 = self.SEM4(e4, r)
        return s1, s2, s3, s4
    
class Decoder(nn.Module):
    # Decoder for extracting difference information
    def __init__(self, channels, bilinear=False):
        super(Decoder, self).__init__()
        self.up5_4 = up(channels[4], channels[3], bilinear=bilinear, scale=2)
        self.D4 = BaseModule(channels[4],channels[3])
        self.up4_3 = up(channels[3], channels[2], bilinear=bilinear, scale=2)
        self.D3 = BaseModule(channels[3],channels[2])
        self.up3_2 = up(channels[2], channels[1], bilinear=bilinear, scale=2)
        self.D2 = BaseModule(channels[2],channels[1])
        self.up2_1 = up(channels[1], channels[0], bilinear=bilinear, scale=2)
        self.D1 = BaseModule(channels[1],channels[0])
        
    def forward(self, s1, s2, s3, s4, r):

        # Similar to UNet
        d4 = self.D4(torch.cat([self.up5_4(r),s4],1))
        d3 = self.D3(torch.cat([self.up4_3(d4),s3],1))
        d2 = self.D2(torch.cat([self.up3_2(d3),s2],1))
        d1 = self.D1(torch.cat([self.up2_1(d2),s1],1))
        return d1, d2, d3, d4

class WCM(nn.Module):
    # A weighted classification module
    def __init__(self, channels, channel, scale=1):
        super(WCM, self).__init__()
        '''
        channel: Number of channels of the input feature map di
        scale: The scale of up-sampling of the generated prediction map
        '''
        self.gamma = nn.Conv2d(channel, 2, kernel_size=1, padding=0, bias=False)
        self.delta = nn.Conv2d(channels[4], 2, kernel_size=1, padding=0, bias=False)
        self.up = nn.Sequential() if scale >= 2 else nn.Upsample(scale_factor=scale, mode='bilinear',align_corners=True)
    
    def forward(self, di, vector):
        weight = self.delta(vector)
        pi = self.gamma(di) * weight
        map = self.up(pi)
        return map

class all_WCM(nn.Module):
    # Combination of all weighted classification modules
    def __init__(self, channels):
        super(all_WCM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.WCM1 = WCM(channels, channels[0])
        self.WCM2 = WCM(channels, channels[1], 2)
        self.WCM3 = WCM(channels, channels[2], 4)
        self.WCM4 = WCM(channels, channels[3], 8)

    def forward(self, d1, d2, d3, d4, r):
        vector = self.avg_pool(r)
        map1 = self.WCM1(d1, vector)
        map2 = self.WCM2(d2, vector)
        map3 = self.WCM3(d3, vector)
        map4 = self.WCM4(d4, vector)
        _m = map1 + map2 + map3 + map4
        return _m

class SESNet(nn.Module):
    def __init__(self,image_channels, init_channels, bilinear):
        super(SESNet, self).__init__()
        channels = [init_channels, init_channels*2, init_channels*4, init_channels*8, init_channels*16, init_channels*32]
        
        '''
        image_channels: Number of channels of the input image
        init_channels: Number of channels in the initial feature map
        channels: regulates the overall number of channels in the network, which is the "C" in the paper
        bilinear: Up-sampling method of the decoder (default deconvolution)
        '''

        self.Encoder = Encoder(channels, image_channels)
        self.get_featureMap_r = get_featureMap_r(channels)
        self.all_SEM = all_SEM(channels)
        self.Decoder = Decoder(channels,bilinear)
        self.all_WCM = all_WCM(channels)

    def forward(self, I1, I2):
        x1, x2, x3, x4 = self.Encoder(I1, I2)
        r = self.get_featureMap_r(x4)
        x1,x2,x3,x4 = self.all_SEM(x1,x2,x3,x4,r)
        x1,x2,x3,x4 = self.Decoder(x1,x2,x3,x4,r)
        x = self.all_WCM(x1,x2,x3,x4,r)
        return x, r
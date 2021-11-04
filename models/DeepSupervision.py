import torch.nn as nn
from utils.parser import get_parser_with_args
from utils.helpers import get_criterion

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
criterion = get_criterion(opt)

'''
_m: Output of the main network structure
r : Output of node R
_r: To achieve deep supervision, r is fed into a 1×1 convolutional layer followed by bilinear upsampling
loss_main: Loss of main network structure
loss_DS: Loss of deep supervision
'''

class DSLoss(nn.Module):
    # Loss function for deep supervision
    def __init__(self):
        super(DSLoss, self).__init__()
        self.conv = nn.Conv2d(opt.init_channels*16, 2, kernel_size=1, padding=0, bias=False)
        self.up = nn.Upsample(scale_factor=16, mode='bilinear',align_corners=True)
        
    def forward(self, r, labels):
        _r = self.up(self.conv(r))
        loss_DS = criterion(_r, labels)
        return loss_DS
    
class DSNet(nn.Module):
    def __init__(self, net):
        super(DSNet, self).__init__()
        self.net = net
        self.DSloss = DSLoss()
        
    def forward(self, I1, I2, labels):

        # Get the output of the main network and node R
        _m, r = self.net(I1, I2)

        # _m and _r are supervised by labels
        loss_main = criterion(_m, labels)
        loss_DS = self.DSloss(r, labels)
        return loss_main, loss_DS
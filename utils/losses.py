from utils.parser import get_parser_with_args
from utils.metrics import dice_loss, FocalLoss
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def WBCE(w1=0.45,w2=0.55):
    weight_CE = torch.FloatTensor([w1,w2]).to(dev)
    return nn.CrossEntropyLoss(weight=weight_CE)

def hybrid_loss(predictions, target):  # batch balence
    loss1 = FocalLoss()
    loss2 = dice_loss
    loss = loss1(predictions, target) + loss2(predictions, target)
    return loss

class WCELoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(WCELoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))
            
        SumC = torch.sum(target).float()
        SumU = target.view(-1).shape[0] - SumC
        WeightU = (SumC+SumU)/(2*SumU)
        WeightC = (SumC+SumU)/(2*SumC)
        self.alpha= torch.Tensor([WeightU,WeightC])
        
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
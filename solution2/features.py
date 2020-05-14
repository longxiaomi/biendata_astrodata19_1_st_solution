import torch
import torch.nn as nn
import torch.nn.functional as F
class features(nn.Module):
    def __init__(self):
        super(features, self).__init__()

    def forward(self, x):
        x_ = -torch.unsqueeze(x, 1)+1
        x=torch.unsqueeze(x,1)+1

        x_ = torch.cat([torch.ones_like(x_), x_], 1)
        x = torch.cat([torch.ones_like(x), x], 1)

        x=  torch.log(torch.max(x,1,keepdim=True).values)
        x_ = torch.log(torch.max(x_, 1, keepdim=True).values)

        x0 = torch.avg_pool1d(x,3,1,1)
        x5 = torch.avg_pool1d(x, 199, 1, 99)
        x0_ = torch.avg_pool1d(x_, 3, 1, 1)
        x5_ = torch.avg_pool1d(x_, 199, 1, 99)
        x = torch.cat([x0-x5,x0,x0_-x5_,x0_], dim=1)
        return x

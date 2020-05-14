import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((channel,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, k = x.size()
        y=torch.squeeze(self.avg_pool(x),-1)
        #y=torch.squeeze(F.max_pool1d(x,k,1),-1)
        y = self.fc(y)#.view(b, c, 1, 1)
        return y
class model(nn.Module):
    def __init__(self, seqlenth, featuresize=9, seqembedding=3, dropout=0.2):
        super(model, self).__init__()
        self.lossfun = nn.CrossEntropyLoss()
        #self.lossfun = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.7,1,1],dtype=np.float32)))
        self.seqembedding = nn.Parameter(data=torch.rand([seqembedding, seqlenth]), requires_grad=True)
        # self.lossfun = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.1,1,1],dtype=np.float32)))
        self.bn0 = nn.BatchNorm1d(featuresize + seqembedding)
        self.layer1 = nn.Conv1d(featuresize + seqembedding, 256, 3, 1, padding=0)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Conv1d(256, 256, 3, 1, padding=0)
        self.bn2 = nn.BatchNorm1d(256)

        self.layer3 = nn.Conv1d(256, 256, 5, 1, padding=0)
        self.bn3 = nn.BatchNorm1d(256)
        self.layer4 = nn.Conv1d(256, 256, 5, 1, padding=0)
        self.bn4 = nn.BatchNorm1d(256)

        self.layer5 = nn.Conv1d(256, 256, 7, 1, padding=0)
        self.bn5 = nn.BatchNorm1d(256)
        self.layer6 = nn.Conv1d(256, 256, 7, 1, padding=0)
        self.bn6 = nn.BatchNorm1d(256)

        self.layer7 = nn.Conv1d(256, 256, 9, 1, padding=0)
        self.bn7 = nn.BatchNorm1d(256)
        self.layer8 = nn.Conv1d(256, 256, 9, 1, padding=0)
        self.bn8 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(1024, 3)
        self.selayer=SELayer(256,16)
        self.selayer2=SELayer(256,16)
        self.selayer3=SELayer(256,16)

        

    def forward(self, x, labels=None):
        seqembedding = self.seqembedding.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([x, seqembedding], 1)
        x = self.bn0(x)
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.max_pool1d(x, 2, 2)

        y=self.selayer(x)
        x=torch.unsqueeze(y,-1)*x


        x = F.relu(self.bn3(self.layer3(x)))
        x = F.relu(self.bn4(self.layer4(x)))
        x = F.max_pool1d(x, 4, 4)
        
        y2=self.selayer2(x)
        x=torch.unsqueeze(y2,-1)*x

        x = F.relu(self.bn5(self.layer5(x)))
        x = F.relu(self.bn6(self.layer6(x)))
        x = F.max_pool1d(x, 6, 6)

        y3=self.selayer3(x)
        x=torch.unsqueeze(y3,-1)*x

        x = F.relu(self.bn7(self.layer7(x)))
        x = F.relu(self.bn8(self.layer8(x)))
        x = F.max_pool1d(x, 8, 8)

        x = torch.reshape(x, [x.shape[0], -1])

        x = self.dropout(x)
        logist = self.fc1(x)
        if labels is not None:
            #one_hot = torch.zeros(labels.shape[0], 3).cuda().scatter_(1, labels, 1)
            #loss2 = torch.mean(one_hot[:, 1] * torch.softmax(logist,-1)[:, 2],-1)
            #loss1 = torch.mean(one_hot[:, 2] * torch.softmax(logist,-1)[:, 1],-1)
            loss = self.lossfun(logist, labels)
            return logist, loss
        return logist

    def inference(self, x):
        logist = self.forward(x)
        return list(torch.argmax(logist, dim=1).cpu().numpy()), list(
            torch.softmax(logist, dim=1).cpu().detach().numpy())


def save(model, step, outputdir, MaxModelCount=5):
    checkpoint = []
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    if os.path.exists(os.path.join(outputdir, 'checkpoint')):
        with open(os.path.join(outputdir, 'checkpoint')) as f:
            checkpoint = f.readlines()
    checkpoint.append('model_' + str(step) + '.kpl\n')
    logging.info('Saving model as \"' + checkpoint[-1].strip('\n') + '\"')
    while len(checkpoint) > MaxModelCount:
        os.remove(os.path.join(outputdir, checkpoint[0].strip('\n')))
        checkpoint.pop(0)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
    }, os.path.join(outputdir, checkpoint[-1].strip('\n')))
    with open(os.path.join(outputdir, 'checkpoint'), 'w') as f:
        f.writelines(checkpoint)


def load(model, outputdir):
    checkpoint = []
    if os.path.exists(os.path.join(outputdir, 'checkpoint')):
        with open(os.path.join(outputdir, 'checkpoint')) as f:
            checkpoint = f.readlines()
    if len(checkpoint) < 1:
        return model, 1, 0
    modelpath = os.path.join(outputdir, checkpoint[-1].strip('\n'))
    logging.info('Restoring model from \"' + checkpoint[-1].strip('\n') + '\"')
    dic = torch.load(modelpath, map_location='cpu')
    model.load_state_dict(dic['model_state_dict'])
    step = dic['step']
    return model, step

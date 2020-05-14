# -*- coding: utf-8 -*-
"""
Created on 2020-4-17

@author: zhangboshen
"""

'''
  数据不平衡较为严重，star占比0.839, galaxy占比0.122， qso占比0.04
'''

import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import logging
import time
import model
from sklearn.metrics import f1_score
from scipy.interpolate import interp1d
from sklearn.model_selection import StratifiedKFold,KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_classes = 3
batch_size = 256
nepoch = 20
learning_rate = 0.001

# Data Aug
randomCropRatio = 0.2
randomPointRatio = 0.8
randomseed = 100
k_fold = 10 #折数
RootDir = '/data/data1/zhangboshen/CODE/AstroData/'
save_dir = RootDir + 'Solution1/weights/Fold_%d_CNN_all_vlad/'%k_fold

try:
    os.makedirs(save_dir)
except OSError:
    pass


timer = time.time()

train_data = np.load(RootDir + './data/train_data_573417.npy')
train_label = np.load(RootDir + './data/train_label_573417.npy')

print('train_data_shape:',train_data.shape)

test_data = np.load(RootDir + './data/test_data_190624.npy')
test_label = np.load(RootDir + './data/test_label_190624.npy')

print('test_data_shape:',test_data.shape)

## 合并数据训练
train_data = np.concatenate((train_data,test_data), axis=0)
train_label = np.concatenate((train_label,test_label), axis=0)

timer = time.time()-timer
print('load data cost time: ', timer)

class myDalaLoader(torch.utils.data.Dataset):
    def __init__(self, DATA, LABEL, isAug=True):
        self.DATA = DATA
        self.LABEL = LABEL
        self.isAug = isAug
        
    def __getitem__(self, index):
        data = self.DATA[index]
        label = self.LABEL[index]

        if self.isAug:
            ##### dataAug1: 随机裁剪0.8倍的窗口，resize到2600送进去训练
            ##### https://tianchi.aliyun.com/forum/postDetail?spm=5176.12586969.1002.90.7e036693IfdY9s&postId=4914
            #start = random.randint(0, 2600*randomCropRatio)
            #data_crop = data[start:]   
            #interpFunc = interp1d(np.linspace(0, 2600-1, num=len(data_crop)), data_crop, kind='cubic')
            #data = np.array(interpFunc(np.linspace(0, 2600-1, num=2600)))

            ##### dataAug2: 随机取出来num_points个点, 然后resize为2600长度送进去训练
            num_points = random.randint(2600*randomPointRatio, 2600)
            index = np.random.choice(2600, num_points, replace=False) # False表示无放回采样
            index.sort()
            data_sampled = data[index]   
            interpFunc = interp1d(np.linspace(0, 2600-1, num=len(data_sampled)), data_sampled, kind='cubic')
            data = np.array(interpFunc(np.linspace(0, 2600-1, num=2600)))

        data = np.expand_dims(data, axis = 0)
        data = np.expand_dims(data, axis = 2)
        label = torch.from_numpy(np.array(label)) 
        data = torch.from_numpy(np.array(data))  
        return data, label
            
    def __len__(self):
        return len(self.DATA)


def train_phase(netR, fold, trainDataloaders, valDataloaders, train_index, eval_index):
	# 数据不平衡较为严重，star占比0.839, galaxy占比0.122， qso占比0.04
	weight = np.array([1,1,1]) 
	weight = torch.from_numpy(weight).float()
	criterion = nn.CrossEntropyLoss(weight = weight).cuda()
	optimizer = optim.Adam(netR.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-6)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.2)
	logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                    filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
	logging.info('======================================================__batchsize:%d'%batch_size)
	max_val_f1 = 0
	## Traininig
	for epoch in range(nepoch):
	    print('======>>>>> Online fold: #%d, epoch: #%d, lr=%f<<<<<======' %(fold, epoch, scheduler.get_lr()[0]))
	    # switch to train mode
	    netR.train()
	    train_loss_add = 0.0
	    for step, data in enumerate(trainDataloaders):
	        inputs, label = data
	        img = Variable(inputs.cuda().float())
	        label = Variable(label.cuda().float())
	        # compute output
	        optimizer.zero_grad()
	        estimation = netR(img)
	        loss = criterion(estimation, label.long())
	        # compute gradient and do SGD step
	        loss.backward()
	        optimizer.step()
			
	        train_loss_add = train_loss_add + (loss.item())

	        if(step % 200 == 0 ):
	            print('fold: %d, epoch: %d,  step: %d, loss: %f'%(fold, epoch, step, loss.item()))
	    scheduler.step(epoch)
	    train_loss_add = train_loss_add / len(train_index)
	    print('mean error of 1 sample: %f, #train_indexes = %d' %(train_loss_add, len(train_index)))
		
	    # switch to evaluate mode
	    test_loss_add = 0.0
	    netR.eval()
	    test_loss_add = 0.0
	    pred_result = torch.FloatTensor()
	    with torch.no_grad():
		    for step, data in enumerate(valDataloaders):
		        inputs, label = data
		        img = Variable(inputs.cuda().float())
		        label = Variable(label.cuda().float())
		        estimation = netR(img)	
		        
		        loss = criterion(estimation, label.long())
		        pred_result = torch.cat((pred_result, torch.sigmoid(estimation).data.cpu()), 0)
		        test_loss_add = test_loss_add + (loss.item())
		        if(step % 200 == 0 ):
		            print('Evaluating fold_%d epoch_%d,  step: %d, loss: %f'%(fold, epoch, step, loss.item()))            

	    test_loss_add = test_loss_add / len(eval_index)
	    print('mean error of 1 sample: %f, #test_indexes = %d' %(test_loss_add, len(eval_index)))

	    # F1 score
	    val_predictions = [np.argmax(value) for value in pred_result]
	    val_GT = train_label[eval_index]
	    val_f1 = f1_score(val_GT, val_predictions, average='macro')
	    print('------------fold_%d, epoch_%d, f1_score:'%(fold, epoch), val_f1)
	    each_val_f1 = f1_score(val_GT, val_predictions, average=None)
	    print('------------fold_%d, epoch_%d, each class f1_score:'%(fold, epoch), each_val_f1[0],each_val_f1[1],each_val_f1[2])
	    if val_f1 > max_val_f1:
	        max_val_f1 = val_f1
	        torch.save(netR.state_dict(), '%s/netR_fold_%d_val_f1_%.4f.pth' % (save_dir, fold, max_val_f1))
	    # log
	    logging.info('Fold %d_Epoch %d: train_error=%.5f, test_error=%.5f, valF1=%.5f, eachClassF1=%.5f_%.5f_%.5f, best_val_f1=%.5f, lr = %.6f'
	    %(fold, epoch, train_loss_add, test_loss_add, val_f1, each_val_f1[0],each_val_f1[1],each_val_f1[2],max_val_f1,scheduler.get_lr()[0]))
	print('val best f1:',max_val_f1)
	return max_val_f1
	
def test_phase_oneEpoch(fold, max_val_f1,testDataloaders, test_GT):
    netR.load_state_dict(torch.load('%s/netR_fold_%d_val_f1_%.4f.pth' % (save_dir, fold, max_val_f1)))
    netR.eval()
    out = torch.FloatTensor()
    with torch.no_grad():
        for step, data in enumerate(testDataloaders):
            inputs, label = data
            img = Variable(inputs.cuda().float())
            label = Variable(label.cuda().float())
            estimation = netR(img)  
            estimation = torch.sigmoid(estimation)
            out = torch.cat((out,estimation.data.cpu()), 0)
    
    pred_result = np.array(out)
    test_predictions = [np.argmax(value) for value in pred_result]
    test_f1 = f1_score(test_GT, test_predictions, average='macro')
    print('-------------------------fold_%d, test_set, f1_score:'%fold, test_f1)
    return pred_result, test_f1

if __name__ == '__main__':
    random.seed(randomseed)
    np.random.seed(randomseed)
    torch.manual_seed(randomseed)

    pred_result_train = np.zeros((len(train_data), num_classes))  

    max_val_f1_list = []

    stratified_folder = KFold(n_splits=k_fold, random_state=randomseed, shuffle=True)
    for fold, (train_index, eval_index) in enumerate(stratified_folder.split(train_data)):
        ## 我们用到了下面的8个模型
        #netR = model.CNN_moreconv_vlad_last(num_classes = num_classes)
        #netR = model.CNN_vlad_last(num_classes = num_classes)
        #netR = model.CNN_vlad_last_12_branch(kernel_size_list=[3,5,7,9,11,13,15,17,21,25,31,41], num_classes=num_classes)
        #netR = model.CNN_vlad_last_SE(num_classes = num_classes)
        #netR = model.CNN_vlad_last_SE_V3(num_classes = num_classes)
        #netR = model.CNN_all_vlad(num_classes = num_classes)
        netR = model.CNN_residual_vlad_last(num_classes = num_classes)
        #netR = model.CNN_residual_vlad_last_10_branch(num_classes = num_classes)

        netR.cuda()
        image_datasets_train = myDalaLoader(train_data[train_index], train_label[train_index], isAug=False)    
        trainDataloaders = torch.utils.data.DataLoader(image_datasets_train, batch_size = batch_size,
                                           shuffle=True, num_workers=8, drop_last=True)
        image_datasets_val = myDalaLoader(train_data[eval_index], train_label[eval_index], isAug=False)    
        valDataloaders = torch.utils.data.DataLoader(image_datasets_val, batch_size = batch_size,
                                           shuffle=False, num_workers=8, drop_last=False)

        max_val_f1 = train_phase(netR, fold, trainDataloaders, valDataloaders, train_index, eval_index)
        max_val_f1_list.append(max_val_f1)

        pred_logits_val, _ = test_phase_oneEpoch(fold, max_val_f1, valDataloaders, train_label[eval_index])
        pred_result_train[eval_index] = pred_logits_val

    #np.save(save_dir+'netR_fold_%d_ensemble_trainSet.npy'%k_fold, pred_result_train/k_fold)
    print('trainset f1 score after k-fold ensemble:', f1_score(train_label, [np.argmax(v) for v in pred_result_train], average='macro'))
    max_val_f1_list = np.array(max_val_f1_list)
    np.save(save_dir+'max_val_f1_list.npy', max_val_f1_list)



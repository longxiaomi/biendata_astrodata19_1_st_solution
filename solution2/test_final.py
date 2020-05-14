import logging
import numpy as np
import random
import datahelper
import torch
import model
import trainer
import os
import torch.utils.data as Data
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
import torch.nn.functional as F
import sklearn
import features


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

testingImagNumber = 190624 #190624 
n_class = 3

save_dir = '/data/data1/zhangboshen/CODE/AstroData/2D_conv/save_weights/with_valdata'

seqlenth = 2600
batchsize = 128
UsingGPU=True
K =10

## 获取测试集图片名
test_data = np.load('/data/data1/zhangboshen/CODE/AstroData/data/test_sets/test_set_data.npy')

test_data_ = torch.from_numpy(test_data)
# test_label_ = torch.from_numpy(test_label)
test_dataset = Data.TensorDataset(test_data_.float())
test_loader = Data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=test_dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=2,
)

logging.basicConfig(level=logging.INFO,
                    # filename=save_dir + '/log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

mymodel =model.model(seqlenth,featuresize=4,seqembedding=3,dropout=0.5)
featuremodel=features.features()
if UsingGPU:
    mymodel=mymodel.cuda()
    featuremodel=featuremodel.cuda()

estimation = np.zeros((testingImagNumber, n_class))
for i in range(K):
    # logger.info('*****{}***** fold test start.'.format(i))
    model_list = os.listdir(save_dir+'/Fold_{}'.format(i))
    model_list.sort()
    # print(save_dir+'/baseline_{}/'.format(i)+model_list[-1])
    print(save_dir+'/Fold_{}/'.format(i)+model_list[-1])
    temp_model = torch.load(save_dir+'/Fold_{}/'.format(i)+model_list[-1])
    mymodel.load_state_dict(temp_model['model_state_dict'])
    mymodel.eval()
    out = torch.FloatTensor()
    pre_labels=[]
    for batch in test_loader:
        _features = batch[0]
        if UsingGPU:
            _features = _features.cuda()
            # labels = labels.cuda()
        logist=mymodel(featuremodel(_features))
        pre_labels =pre_labels + list(torch.argmax(logist, dim=1).cpu().numpy())
        logist = torch.sigmoid(logist)
        out = torch.cat((out,logist.data.cpu()), 0)

    # precision,recall,f1,_=sklearn.metrics.precision_recall_fscore_support(ground_true,pre_labels,labels=range(n_class),average='macro')
    # logger.info("fold:{}, precision:{:.5f}, recall:{:.5f}, f1:{:.5f}".format(i,precision, recall, f1))
    logger.info("fold:{} complete.".format(i))
    estimation = estimation + out.numpy()

estimation = estimation/K
np.save(save_dir+'/ext_1_pred_9844.npy',estimation)
pre_label_all = list(np.argmax(estimation, axis = 1))
np.save(save_dir+'/_pre_label.npy',pre_label_all)
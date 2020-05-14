import logging
import numpy as np
import random
import datahelper
import torch
import model_se as model
import trainer_se as trainer
import os
import torch.utils.data as Data
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO,
                    # filename='./log/log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# datafile='./train.mymemmap'#train.csv文件位置
datafile = '/data/data1/zhangboshen/CODE/AstroData/data/train_data_573417.npy'
labelfile = '/data/data1/zhangboshen/CODE/AstroData/data/train_label_573417.npy'
num_labels=3
batchsize=200
output_dir='./save_weights/with_valdata_se/Fold'#模型保存目录
UsingGPU=True
randomseed=5
seqlenth=2600
epoch=10
lr=1e-3
k_fold=10#折数
if __name__ == '__main__':
    random.seed(randomseed)
    np.random.seed(randomseed)
    torch.manual_seed(randomseed)
    if UsingGPU:
        torch.cuda.manual_seed_all(randomseed)
    #train_rows,labels =datahelper.getdata(datafile,range(0,56),True)
    # train_rows,labels =datahelper.getnumpydata(datafile, labelfile)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train_rows = np.load(datafile)
    labels = np.load(labelfile)
    print(train_rows.shape,labels.shape)
    test_data = np.load('/data/data1/zhangboshen/CODE/AstroData/data/test_data_190624.npy')
    test_label = np.load('/data/data1/zhangboshen/CODE/AstroData/data/test_label_190624.npy')
    train_rows = np.concatenate((train_rows,test_data), axis = 0)
    labels = np.concatenate((labels,test_label), axis = 0)
    print(train_rows.shape,labels.shape)
    # print('total: {}'.format(len(labels)))
    stratified_folder = KFold(n_splits=k_fold, random_state=randomseed, shuffle=True)
    for k, (train_index, eval_index) in enumerate(stratified_folder.split(train_rows)):
        train_temp = train_rows[train_index,:]
        label_temp = labels[train_index]
        # for i in train_index:
        #     train_temp.append(train_rows[i])
        #     label_temp.append(labels[i])
        train_data = torch.from_numpy(train_temp)
        train_labels = torch.from_numpy(label_temp)
        train_dataset = Data.TensorDataset(train_data.float(),train_labels.long())
        train_loader = Data.DataLoader(
            # 从数据库中每次抽出batch size个样本
            dataset=train_dataset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=2,
        )
        print('train data finish')
        eval_data = torch.from_numpy(np.array(train_rows[eval_index,:]))
        eval_labels = torch.from_numpy(labels[eval_index])
        eval_dataset = Data.TensorDataset(eval_data.float(), eval_labels.long())
        eval_loader = Data.DataLoader(
            # 从数据库中每次抽出batch size个样本
            dataset=eval_dataset,
            batch_size=batchsize,
            shuffle=False,
            num_workers=2,
        )

        logger.info("Starting{}-fold".format(k))
        num_train_steps = int(len(train_index) * epoch / batchsize)
        mymodel =model.model(seqlenth,featuresize=4,seqembedding=3,dropout=0.5)
        optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr)
        trainer.train(mymodel, optimizer, output_dir+'_'+str(k), epoch, train_loader, eval_loader, UsingGPU=UsingGPU,
                      min_f1score=0.98, maxtokeep=3, CVAfterEpoch=1,classnum=num_labels)





import logging
import numpy as np
import random
from pathlib import Path
import torch
import sklearn
import features
import model
logging.basicConfig(level=logging.INFO,
                    # filename='./log/log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
def train(mymodel,myoptimizer,output_dir,epoch,train_dataloader,eval_dataloader,UsingGPU=True,min_f1score=0.8,maxtokeep=3,CVAfterEpoch=2,classnum=3):
    featuremodel=features.features()
    if UsingGPU:
        mymodel=mymodel.cuda()
        featuremodel=featuremodel.cuda()
    num_train_steps = int(epoch*len(train_dataloader.dataset) / train_dataloader.batch_size)
    logger.info("***** Do train *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Batch size = %d", train_dataloader.batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    global_step = 0  # 共迭代的次数
    maxf1score=min_f1score
    for i in range(1,epoch+1):
        logger.info("********epoch:{}********".format(i))
        for p in myoptimizer.param_groups:
            p['lr'] = p['lr'] * 0.8
        for batch in train_dataloader:
            global_step += 1
            _features,labels = batch
            if UsingGPU:
                _features = _features.cuda()
                labels = labels.cuda()
            logist,loss=mymodel(featuremodel(_features),labels)
            loss.backward()
#             fgm.attack()  # 在embedding上添加对抗扰动
#             logist,loss_adv = mymodel(input_ids, segment_ids, input_mask, label_ids)
#             loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#             fgm.restore()  # 恢复embedding参数
            myoptimizer.step()
            myoptimizer.zero_grad()
            if global_step%100==0:
                logger.info("step:{}, loss:{:.5f}".format(global_step,loss.data))
            if global_step%500==0 and i>=CVAfterEpoch:
                mymodel.eval()
                precision, recall, f1=eval(mymodel,eval_dataloader,classnum,UsingGPU)
                mymodel.train()
                logger.info("step:{}, precision:{:.5f}, recall:{:.5f}, f1:{:.5f}".format(global_step,precision, recall, f1))
                if f1>maxf1score:
                    maxf1score=f1
                    model.save(mymodel,global_step, output_dir, MaxModelCount=maxtokeep)
def eval(mymodel,eval_dataloader,classnum,UsingGPU):
    pre_labels=[]
    ground_true=[]
    for batch in eval_dataloader:
        _features, labels = batch
        featuremodel = features.features()
        if UsingGPU:
            _features = _features.cuda()
            labels = labels.cuda()
            featuremodel=featuremodel.cuda()
        pre_label,_=mymodel.inference(featuremodel(_features))
        pre_labels =pre_labels+ pre_label
        ground_true=ground_true+list(labels.cpu().numpy())
    precision,recall,f1,_=sklearn.metrics.precision_recall_fscore_support(ground_true,pre_labels,labels=range(classnum))
    return sum(precision)/3,sum(recall)/3,sum(f1)/3


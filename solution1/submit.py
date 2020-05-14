
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

model_dict = {   
1: 'CNN_all_vlad_0.9837_fold_10_ensemble_testSet.npy',           
2: 'CNN_vlad_last_0.9839_fold_10_ensemble_testSet.npy',          
3: 'CNN_residual_vlad_last_0.9841_fold_10_ensemble_testSet.npy', 
4: 'CNN_moreconv_vlad_last_0.9837_fold_10_ensemble_testSet.npy', 
5: 'CNN_vlad_last_SE_0.9836_fold_10_ensemble_testSet.npy',       
6: 'ext_1_pred_9844.npy',                                        
7: 'CNN_vlad_last_SE_V3_0.9843_fold_10_ensemble_testSet.npy',    
8: 'CNN_vlad_last_12_branch_0.9839_fold_10_ensemble_testSet.npy',
9: 'CNN_residual_vlad_last_10_branch_0.9832_fold_10_ensemble_testSet.npy', 
10: 'ext_2_pred_9843.npy'                                        
}

test_ids = np.load('/data/data1/zhangboshen/CODE/AstroData/data/test_set_zbs/test_set_ids.npy')
RootDir = './our_result/'

weight = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

print(weight, 'sum(weight)', sum(weight))

final_logits = np.zeros((190624, 3))

for i in range(len(model_dict)):
    tmp = np.load(RootDir + model_dict[i+1])
    final_logits += weight[i]*tmp

np.save(RootDir + 'final_submission.npy', final_logits)

test_csv = pd.DataFrame()
test_csv['id'] = test_ids
test_csv['label'] = [np.argmax(v) for v in final_logits]    
test_csv['label'] = test_csv['label'].map({0:'star', 1:'galaxy', 2:'qso'})
test_csv.to_csv(RootDir + 'final_submission.csv', index=False)

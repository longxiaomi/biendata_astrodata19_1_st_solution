import numpy as np
import pandas as pd
from tqdm import tqdm



# 统计一下数据有多少

def countFiles(file):
    count=0
    f = open(file)
    line = f.readline()
    while line:
        count += 1
        line = f.readline()
    f.close()
    print(file, count-1)  



test_set_data = np.zeros((20000*9 + 10624, 2600))
test_set_ids = []

start_index = 0
for i in tqdm(range(10)):
    file = '/data/data1/zhangboshen/CODE/AstroData/data/test_set_zbs/test_sets_%d.csv'%i    # 9个20000，一个10624
    #countFiles(file)
    pd_file = pd.read_csv(file)
    test_set_ids.extend(list(pd_file['id']))
    pd_file_data = pd_file.drop(columns=['id'])
    print(file, pd_file_data.shape)

    test_set_data[start_index : start_index+len(pd_file)] = np.array(pd_file_data)

    start_index += len(pd_file)


np.save('test_set_data.npy', test_set_data)
np.save('test_set_ids.npy', np.array(test_set_ids))

 
一共输出两个模型文件： with_valdata 和 with_valdata_se
1. with_valdata 
训练
python train.py
测试（储存每个样本属于每一类的权重）
python test_final.py

2. with_valdata_se
训练
python train_se.py
测试（储存每个样本属于每一类的权重）
python test_final_se.py

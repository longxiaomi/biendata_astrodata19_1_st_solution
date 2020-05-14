# biendata_astrodata19_1_st_solution

这是Biendata中的天文数据分类竞赛第一名的解决方案，竞赛网址为：https://www.biendata.com/competition/astrodata2019/

队伍名：星星你个星星


### 代码说明

请参考`代码说明.txt`文件，运行环境主要依赖Pytorch1.2，完整依赖位于`requirements.txt`


### 一些实验结论

#### working:

- 1D CNN很适合该任务，感受野很重要（多尺度kernel size）;
- 一些网络结构，包括SE、VLAD等；
- K折交叉验证+朴素集成。

#### Not working:

- BN/dropout；
- 数据重采样/focal loss/类别加权交叉熵损失；
- Lookahead优化器；
- 均值归一化/min_max归一化；
- F1 loss.


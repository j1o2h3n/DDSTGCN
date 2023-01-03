# DDSTGCN

Y. Sun, X. Jiang, Y. Hu, F. Duan, K. Guo, B. Wang, J. Gao, B. Yin, "[Dual Dynamic Spatial-Temporal Graph Convolution Network for Traffic Prediction](https://ieeexplore.ieee.org/document/9912360)," in IEEE Transactions on Intelligent Transportation Systems, 2022, doi: 10.1109/TITS.2022.3208943.


## Requirements

- python 3
- numpy
- torch

## Train Commands

```
python train.py
```
To run different datasets, you need to modify the relevant parameters of the dataset, including '--data', '--adjdata', '--in_dim', '--num_nodes'. The default is METR-LA dataset.


### Dataset

<!--六个数据集基本信息--data --adjdata --in_dim --num_nodes-->
Dataset URL: https://drive.google.com/drive/folders/1uoY8ROQU73BqWyl566ZNdRBOOTM4T2DS?usp=sharing

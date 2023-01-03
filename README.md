# DDSTGCN

Y. Sun, X. Jiang, Y. Hu, F. Duan, K. Guo, B. Wang, J. Gao, B. Yin, "[Dual Dynamic Spatial-Temporal Graph Convolution Network for Traffic Prediction](https://ieeexplore.ieee.org/document/9912360)," in IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 12, pp. 23680-23693, Dec. 2022, doi: 10.1109/TITS.2022.3208943.


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


## License
This repo is under [MIT license](LICENSE).


### Citation
If you find this repo useful in your research, please cite the following in your manuscript:

```bibtex
@ARTICLE{sun2022dual,
author={Sun, Yanfeng and Jiang, Xiangheng and Hu, Yongli and Duan, Fuqing and Guo, Kan and Wang, Boyue and Gao, Junbin and Yin, Baocai},
   journal={IEEE Transactions on Intelligent Transportation Systems},
   title={Dual Dynamic Spatial-Temporal Graph Convolution Network for Traffic Prediction},
   year={2022},
   volume={23},
   number={12},
   pages={23680-23693},
   publisher={IEEE},
   doi={10.1109/TITS.2022.3208943}
}
```

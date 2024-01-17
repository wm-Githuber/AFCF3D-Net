# Adjacent-level Feature Cross-Fusion with 3D CNN for Remote Sensing Image Change Detection

Here, we provide the official pytorch implementation of the paper "Adjacent-level Feature Cross-Fusion with 3D CNN for Remote Sensing Image Change Detection".
![Architecture](https://github.com/wm-Githuber/AFCF3D-Net/assets/66511993/9c2681a4-a582-4b73-8133-55f2c5da5dc9)

# Requirements
* python        3.9.12
* numpy         1.23.1
* pytorch       1.12.1
* torchvision   0.13.1

# Dataset Preparation
## Data Structure
"""  
Change detection data set with pixel-level binary labels;  
├─A  
├─B  
├─label  
└─list  
&emsp;&emsp;├─train.txt  
&emsp;&emsp;├─val.txt  
&emsp;&emsp;├─test.txt  
"""  
A: Images of T1 time  
B: Images of T2 time  
label: label maps  
list: contrains train.txt, val.txt, and test.txt. each fild records the name of image paris (XXX.png).  

## Data Download  
WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html  
LEVIR-CD: https://justchenhao.github.io/LEVIR/  
SYSU-CD: https://github.com/liumency/SYSU-CD  

# Training and Testing
train.py  
Test.py

# Quantitative Results
![image](https://github.com/wm-Githuber/AFCF3D-Net/assets/66511993/7612d847-8ccb-422d-9fee-3b567b8082a4)


# Qualitative Results
![SYSU-result](https://user-images.githubusercontent.com/66511993/210714033-e006d556-97d1-47e9-8423-3de7a983f385.png)


# Licence
The code is released for non-commercial and research purposes only. For commercial purposes, please contact the authors.


# Citation
If you find this work interesting in your research, please cite our paper as follow:  
@ARTICLE{YeCD,  
         author={Ye, Yuanxin and Wang, Mengmeng and Zhou, Liang and Lei, Guangyang and Fan, Jianwei and Qin, Yao},  
         journal={IEEE Transactions on Geoscience and Remote Sensing},  
         title={Adjacent-Level Feature Cross-Fusion With 3-D CNN for Remote Sensing Image Change Detection},  
         year={2023},  
         volume={61},  
         number={},  
         pages={1-14},  
         doi={10.1109/TGRS.2023.3305499}}

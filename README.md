# Adjacent-level Feature Cross-Fusion with 3D CNN for Remote Sensing Image Change Detection

Here, we provide the official pytorch implementation of the paper "Adjacent-level Feature Cross-Fusion with 3D CNN for Remote Sensing Image Change Detection".

![Net-Architecture](https://user-images.githubusercontent.com/66511993/210692766-5c698bdd-5077-4e7b-8274-c899f86b3cf9.png)


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

% # Pre-trained models
% The three dataset pre-trained models are available.


# Citation

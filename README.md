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

# Quantitative Results
![image](https://user-images.githubusercontent.com/66511993/210712080-516684e9-0da2-4a7f-b159-a352a3c86e56.png)

# Qualitative Results
![SYSU-result](https://user-images.githubusercontent.com/66511993/210714033-e006d556-97d1-47e9-8423-3de7a983f385.png)

# Pre-trained
The pretrained models can be downloaded at:  
address：https://pan.baidu.com/s/1Zg_zdMyHIa9V_s3TO-o9pw 
password：ntjw

# Licence
The code is released for non-commercial and reseach purposes only. For commercial purposes, please contact the authors.



# Citation
If you find this work interesting in your research, please cite our paper as follow:  
@

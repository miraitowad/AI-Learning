# 2022 研究生 计算机视觉 模型学习笔记与代码
****
![加油](Fight.jpg)
****
## 实验环境
- 设备环境
    - Ubuntu 18.04
    - RTX 2080 Ti (16GB)
- 软件环境
    - python==3.8.6
    - pytorch==1.8
    - scikit-learn==1.2.0
    - opencv-python==4.1.1
    - matplotlib==3.2.2
    - Pillow==7.1.2
    - scipy==1.4.1
    - tqdm==4.64.0
    - pandas==1.1.4
****
## 基础骨干（BackBone）
- VGG（CIFAR100）
- RESNET（CIFAR100）

## 语义分割（Semantic Segmentation）
- FCN（carvana-image）
- U-NET（carvana-image）

## 目标检测（Object Detection）
- R-CNN系列（VOC）
- YOLO系列（COCO）

## 生成对抗网络（Generative Adversarial Networks）
- DCGAN（Deep Convolutional Generative Adversarial Networks）
- TODO: VAE、扩散模型......
## 机器学习（Machine Learning）
- 决策树
- 支持向量机
- 集成算法

****
**TODO：**
- 使用C++部署模型（预计使用pytorch自带转换方法）
- 自定义数据集训练模型（自己动手标注）
- ......
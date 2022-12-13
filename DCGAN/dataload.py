# coding=utf-8
import torch.autograd
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os

def data_load(BATCH_SIZE=16):
# 图像处理过程
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # mnist dataset mnist数据集下载
    mnist = datasets.MNIST(
        root='./data/', train=True, transform=img_transform, download=True
    )

    # data loader 数据载入
    dataloader = torch.utils.data.DataLoader(
        dataset=mnist, batch_size=BATCH_SIZE, shuffle=True
    )
    
    return dataloader

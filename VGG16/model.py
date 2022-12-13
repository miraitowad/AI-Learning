'''
Pytorch实现-VGG系列模型( 包含11、13、16、19[BN层] )
调用函数vgg-xx()实例化模型
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG模型接口
    参数:
        config:list[]  定义特征提取层的基本结构
        bn:bool  定义是否启用批归一化策略
        N_class:int 类别个数
    '''
    def __init__(self, config, N_class=10, bn=False):
        super(VGG, self).__init__()
        layers = []
        in_channels = 3 # 定义初始卷积核个数
        for v in config:
            if v == 'M': # 最大池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) #卷积层（输入通道数，输出通道数，卷积核大小，零填充）
                # 判断是否添加批归一化层，激活函数也在此添加
                if bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features_extract = nn.Sequential(*layers)
        # 全连接神经网络作为分类器输出预测结果
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, N_class),
        )
        # 初始化卷积核参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features_extract(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



    


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(n):
    """VGG 11-layer model (configuration "A")"""
    return VGG(cfg['11'], n)


def vgg11_bn(n):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(cfg['11'], n, batch_norm=True)


def vgg13(n):
    """VGG 13-layer model (configuration "B")"""
    return VGG(cfg['13'], n)


def vgg13_bn(n):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(cfg['13'], n, batch_norm=True)


def vgg16(n):
    """VGG 16-layer model (configuration "D")"""
    return VGG(cfg['16'], n)


def vgg16_bn(n):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(cfg['16'], n, batch_norm=True)


def vgg19(n):
    """VGG 19-layer model (configuration "E")"""
    return VGG(cfg['19'], n)


def vgg19_bn(n):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(cfg['19'], n, batch_norm=True)
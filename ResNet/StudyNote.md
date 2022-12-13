# RESNET学习笔记
****
## 一、模型简介
ResNet（Residual Neural Network）由微软研究院的Kaiming He等四名华人提出，通过使用ResNet Unit成功训练出了152层的神经网络，并在ILSVRC2015比赛中取得冠军，在top5上的错误率为3.57%，同时参数量比VGGNet低，效果非常突出。ResNet的结构可以极快的加速神经网络的训练，模型的准确率也有比较大的提升。同时ResNet的推广性非常好，甚至可以直接用到InceptionNet网络中。

>论文链接: https://arxiv.org/abs/1512.03385 
****
## 二、模型背景
在ResNet网络提出之前，传统的卷积神经网络都是通过将一系列卷积层与下采样层进行堆叠得到的。但是当堆叠到一定网络深度时，就会出现两个问题。

1. 梯度消失或梯度爆炸。
2. 退化问题(degradation problem)。

所以ResNet论文提出了residual结构（残差结构）来减轻退化问题，结构如下图所示。
![residual 结构图](https://pic2.zhimg.com/v2-bd3ef7a0905bb8f46191ac70b13e0d71_r.jpg)
****
## 三、残差模块
**残差网络的核心思想是：每个附加层都应该更容易地包含原始函数作为其元素之一**

一个残差块有2条路径 F(x) 和 x ，F(x) 路径拟合残差，可称之为残差路径；x 路径为`identity mapping`恒等映射，可称之为`shortcut`。图中的⊕为`element-wise addition`，要求参与运算的 F(x) 和 x 的尺寸要相同。

**shortcut** 路径大致可以分成 2 种，取决于残差路径是否改变了feature map数量和尺寸,如下图所示：

![residual 结构图](https://pic2.zhimg.com/v2-faddf995bd8ae02c2aca929990c192a5_r.jpg)

- 一种是将输入x原封不动地输出, 称为 `BasicBlock`。（ResNet<50）
- 另一种则需要经过  卷积来升维或者降采样，主要作用是将输出与  路径的输出保持shape一致，对网络性能的提升并不明显, 称为 `Bottleneck` 。（ResNet>=50）

## 四、网络结构
### 4.1、ResNet18
![ResNet18 网络结构图](https://pic3.zhimg.com/v2-15cffb5b081c12f37e9d83ed39f614b2_r.jpg)


**流程理解**
- ResNet18的18层代表的是带有权重的18层，包括卷积层和全连接层，不包括池化层和BN层。
- ResNet18 ，使用的是 BasicBlock。layer1，特点是没有进行降采样，卷积层的 stride = 1，不会降采样。在进行 shortcut 连接时，也没有经过 downsample 层。
- 而 layer2，layer3，layer4 的结构图如下，每个 layer 包含 2 个 BasicBlock，但是第 1 个 BasicBlock 的第 1 个卷积层的 stride = 2，会进行降采样。在进行 shortcut 连接时，会经过 downsample 层，进行降采样和降维。
![ResNet18 网络结构图](https://pic4.zhimg.com/v2-9a2fd03b2ead9c9d39fe6dcf49b28e53_r.jpg)

### 4.2、ResNet50
![ResNet18 网络结构图](https://pic4.zhimg.com/v2-99a3cf9d6bb7aaddbddbdbdde72359f7_r.jpg)


**流程理解**
- 在 layer1 中，首先第一个 Bottleneck 只会进行升维，不会降采样。shortcut 连接前，会经过 downsample 层升维处理。第二个 Bottleneck 的 shortcut 连接不会经过 downsample 层。
- 而 layer2，layer3，layer4 的结构图如下，每个 layer 包含多个 Bottleneck，但是第 1 个 Bottleneck 的  卷积层的 stride = 2，会进行降采样。在进行 shortcut 连接时，会经过 downsample 层，进行降采样和降维。
****
## 四、模型优缺点
ResNet采用网络中增加残差网络的方法，解决网络深度增加到一定程度，更深的网络堆叠效果反而变差的问题。在网络深度到一定程度，误差升高，效果变差，梯度消失现象越明显，后向传播时无法把梯度反馈到前面网络层，前面网络参数无法更新，导致训练变差。残差网络增加一个恒等映射，跳过本层或多层运算，同时后向传播过程中，下一层网络梯度直接传递给上一层，解决深层网络梯度消失的问题。
- **优点**：使前馈/反馈传播算法顺利进行，结构更加简单；恒等映射增加基本不会降低网络的性能。
- **缺点**：训练时间长。


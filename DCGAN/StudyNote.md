# GAN学习笔记
****
## 一、GAN简介
2014年Ian Goodfellow首次提出Generative adversarial networks (生成对抗网络)简称GANs，生成对抗网络就开始在计算机视觉领域得到广泛应用，成为对有用的视觉任务网络之一，也是如今计算机视觉热点研究领域之一，其已经出现的应用领域与方向如下：
- 图像数据集生成
- 生成人脸照片
- 生成真实化照片
- 生成卡通照片
- 图像翻译
- 文本到图像翻译(Text2Image Translation)
- 图像语义道照片翻译(Semantic-Image2Photo Translation)
- 人脸正面视图生成(Face Frontal View Generation)
- 新姿态生成(Generate New Human Poses)
- 照片到卡通漫画翻译
- 照片编辑
- 人脸年龄化
- 照片融合(Photo Blending)
- 超像素(Super Resolution)
- 照片修复(Photo Inpainting)
- 视频预测(Video Prediction)
- 三维对象生成(3D Object Generation)

GAN网络主要由生成网络与鉴别网络两个部分，生成网络负责生成新的数据实例、鉴别网络负责鉴别生成的数据实例与真实数据之间的差异，从而区别哪些是真实数据、哪些是假数据,如下图所示：
![image.png](http://5b0988e595225.cdn.sohucs.com/images/20191023/a3dff725d26042fc8f9994fcbc1b214d.png)
>论文地址：https://arxiv.org/pdf/1406.2661.pdf
****
## 二、简单栗子
以mnist手写数据集为例、GAN网络工作的流程如下：

>1.生成网络从随机数据开始，生成一张图像
>
>2.生成的图像被输入到鉴别器中、鉴别器判断它与ground truth数据之间的差异
> 
> 3.鉴别网络分别考虑他们真假的可能性

得到两个网络的反馈

> 1.鉴别网络循环反馈数据与ground truth之间的差异
>
> 2.生成网络持续接受鉴别网络的反馈，不断优化生成器网络

图示如下：
![image.png](http://5b0988e595225.cdn.sohucs.com/images/20191023/4b183c5278f84a9498f392f1435fa580.png)

****
## 三、DCGAN    
### 3.1、模型介绍
简单的说DCGAN(Deep Convolutional Generative Adversarial Networks)就是GAN的扩展版本，生成网络与鉴别网络都是基于深度神经网络构成，对细节生成更加的真实。

>论文地址：https://arxiv.org/abs/1511.06434


### 3.2、模型特点

DCGAN能改进GAN训练稳定的原因主要有：
1. 将所有池化层用卷积层代替，以卷积的方法实现下采样和上采样（在G中使用逆卷积），使得其在上/下采样过程中可以自己学习参数。
2. 移去最后的全连接层，常见的例子是分类任务中常见的全局池化层（如全局maxpooling），作者发现它使得模型更稳定但是收敛速度变慢。
3. 使用Batch Normalization使模型更稳定，可以防止生成器G将所有样本崩溃到一个单点（原文是preventing the generator from collapsing all samples to a singlepoint，不太理解），这种现象在原GAN中有出现。但是直接将BN运用到所有层会导致模型不稳定和样本振荡，为了避免这种情况发生，我们不在生成器G的输出层以及判别器D的输入层加入BN.
4. 在生成器G中使用ReLU激活函数（除了最后一层输出层使用Tanh除外），在判别器D中使用leaky ReLU激活。

### 3.3、网络结构
**3.3.1 生成器 G 结构**
![DCGAN G 结构](https://img-blog.csdn.net/20171013204210374?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQW5kcmV3c2V1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

**3.3.2 判别器 D 结构**
![DCGAN D 结构](https://img-blog.csdn.net/20171013204308594?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvQW5kcmV3c2V1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

值得一提的是，他与基本的GAN网络差别就是使用了卷积神经网络,G上采样采用了反卷积，D下采样则是正常的卷积。

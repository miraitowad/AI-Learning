# 目标检测 学习笔记
****
## 一、目标检测基本概念
目标检测（Object Detection）的任务是找出图像中所有感兴趣的目标（物体），确定它们的类别和位置，是计算机视觉领域的核心问题之一。由于各类物体有不同的外观、形状和姿态，加上成像时光照、遮挡等因素的干扰，目标检测一直是计算机视觉领域最具有挑战性的问题。

![目标检测 示例图](https://img-blog.csdnimg.cn/2020112019245715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3llZ2VsaQ==,size_16,color_FFFFFF,t_70#pic_center)


需要解决的核心问题：

（1）分类问题：即图片（或某个区域）中的图像属于哪个类别。

（2）定位问题：目标可能出现在图像的任何位置。

（3）大小问题：目标有各种不同的大小。

（4）形状问题：目标可能有各种不同的形状。
****
## 二、目标检测算法分类
基于深度学习的目标检测算法主要分为两类：Two stage和One stage。

**2.1、Tow Stage**
先进行区域生成，该区域称之为region proposal（简称RP，一个有可能包含待检物体的预选框），再通过卷积神经网络进行样本分类。

任务流程：特征提取 --> 生成RP --> 分类/定位回归。

常见tow stage目标检测算法有：**R-CNN系列**、SPP-Net、和R-FCN等。

**2.2、One Stage**
不用RP，直接在网络中提取特征来预测物体分类和位置。

任务流程：特征提取–> 分类/定位回归。

常见的one stage目标检测算法有：OverFeat、**YOLO系列**、SSD和RetinaNet等。
****

## 三、R-CNN系列
### 3.1、简介
在传统目标检测算法中，大部分采用的方式是滑动窗口+手工设计特征，这种方法虽然简单有效，但是其弊端也很明显，没有针对性地选择检测窗口，导致其时间复杂度高且信息冗余多，对于现代高像素的图像来说更为如此，同时，传统手工设计的特征对于图像多样性的变化健壮性较差，使用场景较为局限。

针对以上问题，Girshick R在2014年提出了采用Region Proposal（区域建议） + CNN的方法来替代传统方法，设计出了R-CNN框架，大幅度的提升了目标检测的准确率和检测速度。

R-CNN的思路是首先将输入的图像通过Selective Search算法[29]预先找出目标可能出现的区域，利用图像中的纹理、颜色、边缘等信息可以生成2000到3000个数量的候选区域，大幅度缩减了检测窗口的数量，降低了窗口冗余度。接着因为CNN要求输入维度一致，所以将生成的候选区域图像缩放归一化后再利用CNN对其进行特征提取。然后将其特征输入到SVM进行目标分类训练。最后使用一个线性回归器对目标候选区域位置进行精修调整。因其分类和定位是分为两阶段进行，故称为两阶段，大致思路步骤图如下图所示：

![R-CNN 示例图](https://pic1.zhimg.com/80/v2-49289a66fb2ab374cbb6f461442bb078_720w.webp)

正是基于这种思想和方法，R-CNN一经提出，就在VOC2012数据集上，将以往最好的mAP值从30%拔高到了53.3%，从而开启了目标检测领域中深度学习的应用热潮。

虽然R-CNN取得了优秀成绩，不再像传统算法一样对图片进行滑窗穷举，从而减少了检测窗口冗余度，但其数千个候选区域都需要进行CNN提取特征 + SVM分类，重叠区域反复计算导致时间复杂度上升，一张图片的完整检测还是需要高达47秒的时间。

>论文地址：https://arxiv.org/abs/1311.2524

### 3.2、Faster R-CNN模型特点
Faster-rcnn引入了RPN网络用来代替 Selective Search选取候选区域。
![R-CNN 示例图](https://img-blog.csdn.net/20180822165007483?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RvbmtleV8xOTkz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Faster R-CNN主要分为四个部分：1.特征提取 2.RPN 3.ROI感兴趣区域 4.分类和回归。

- Conv Layer: 卷积层包括一系列卷积(Conv + Relu)和池化(Pooling)操作，用于提取图像的特征(feature maps)，一般直接使用现有的经典网络模型ZF或者VGG16，而且卷积层的权值参数为RPN和Fast RCNN所共享，这也是能够加快训练过程、提升模型实时性的关键所在。
- Region Proposal Network: RPN网络用于生成区域候选框Proposal，基于网络模型引入的多尺度Anchor，通过Softmax对anchors属于目标(foreground)还是背景(background)进行分类判决，并使用Bounding Box Regression对anchors进行回归预测，获取Proposal的精确位置，并用于后续的目标识别与检测。
- RoI Pooling: 综合卷积层特征feature maps和候选框proposal的信息，将propopal在输入图像中的坐标映射到最后一层feature map(conv5-3)中，对feature map中的对应区域进行池化操作，得到固定大小(7×77×7)输出的池化结果，并与后面的全连接层相连。
- Classification and Regression: 全连接层后接两个子连接层——分类层(cls)和回归层(reg)，分类层用于判断Proposal的类别，回归层则通过bounding box regression预测Proposal的准确位置。

**RPN网络详解**：

Anchor是RPN网络中一个较为重要的概念，传统的检测方法中为了能够得到多尺度的检测框，需要通过建立图像金字塔的方式，对图像或者滤波器(滑动窗口)进行多尺度采样。RPN网络则是使用一个3×3的卷积核，在最后一个特征图(conv5-3)上滑动，将卷积核中心对应位置映射回输入图像，生成3种尺度(scale){128^2, 256^2, 512^2} {128^2, 256^2, 512^2}和3种长宽比(aspect ratio){1:1,1:2,2:1}{1:1,1:2,2:1}共9种Anchor，如下图所示。特征图conv5-3每个位置都对应9个anchors，如果feature map的大小为W×HW×H，则一共有9个anchors，滑动窗口的方式保证能够关联conv5-3的全部特征空间，最后在原图上得到多尺度多长宽比的anchors。

### 3.3、Faster R-CNN网络结构图
![R-CNN 示例图](https://img-blog.csdn.net/20180822165949661?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2RvbmtleV8xOTkz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 3.4、模型总结
在引进了RPL、RPN等思想和机制后，R-CNN系列模型检测速度及精度均有显著提升，由此可见，将任务分为两阶段，分别进行分类和定位，最后进行结果汇总的方式是行之有效的。
但同时，R-CNN系列算法虽在准确率和召回率方面有良好的表现，可其检测速度略显低下，尤其对于每秒可高达60帧的现代高速摄影机或高帧率视频更为笨拙，所以如何保证其准确率的前提下提高其检测速度，也是当今研究学者们的讨论的热点问题。

****
## 四、YOLO系列
### 4.1、模型简介

R-CNN系列算法蓬勃发展的同时，人们逐渐意识到，虽然R-CNN精度上表现不俗，但始终不能满足实时检测的需求，为定位和分类两个任务分别训练两个模型，势必会导致模型参数量增加从而拖慢检测速度，于是一种基于回归的端对端一阶段任务算法问世，最为著名的便是由Joseph Redmon等人提出的YOLO（You Only Look Once）系列算法，后续也出现了如SSD(Single ShotMultiBox Detector)等优秀的基于回归的目标检测算法。

>论文地址: https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1506.02640

### 4.2、模型特点

1. 统一网络:YOLO没有显示求取region proposal的过程。Faster R-CNN中尽管RPN与fast rcnn共享卷积层，但是在模型训练过程中，需要反复训练RPN网络和fast rcnn网络.相对于R-CNN系列的"看两眼"(候选框提取与分类),YOLO只需要Look Once。
2. YOLO统一为一个回归问题，而R-CNN将检测结果分为两部分求解：物体类别（分类问题），物体位置即bounding box（回归问题）。

### 4.3、模型流程
YOLO算法采用一个单独的CNN模型实现end-to-end的目标检测：首先将输入图片resize到448x448，然后送入CNN网络，最后处理网络预测结果得到检测的目标。相比R-CNN算法，其是一个统一的框架，其速度更快，而且YOLO的训练过程也是end-to-end的，流程如下图所示：

![YOLO 示例图](https://img-blog.csdnimg.cn/20200328143228313.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RGQ0VE,size_16,color_FFFFFF,t_70)


作者在YOLO算法中把物体检测（object detection）问题处理成回归问题，用一个卷积神经网络结构就可以从输入图像直接预测bounding box和类别概率。

### 4.4、网络结构
**YOLOV1**

![YOLOV1 网络结构图](https://p9-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/e2a2c7b4bc8e475ab71d0ca8be47e190~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.image?)
**YOLOV5**

![YOLOV5 网络结构图](https://img-blog.csdnimg.cn/408486b724aa4b46b4a785cfd6c92216.png)


### 4.5、YOLOV5流程

1. Input
    - 和YOLOv4一样，对输入的图像进行Mosaic数据增强。Mosaic数据增强的作者也是来自Yolov5团队的成员，通过随机缩放、随机裁剪、随机排布的方式对不同图像进行拼接，如下如所示：

    - 采用Mosaic数据增强方法，不仅使图片能丰富检测目标的背景，而且能够提高小目标的检测效果。并且在BN计算的时候一次性会处理四张图片！

2. Backbone
    - 骨干网路部分主要采用的是：Focus结构、CSP结构。其中 Focus 结构在YOLOv1-YOLOv4中没有引入，作者将 Focus 结构引入了YOLOv5，用于直接处理输入的图片。Focus重要的是切片操作，如下图所示，4x4x3的图像切片后变成2x2x12的特征图。
    - 以YOLOv5s的结构为例，原始608x608x3的图像输入Focus结构，采用切片操作，先变成304x304x12的特征图，再经过一次32个卷积核的卷积操作，最终变成304x304x32的特征图。

3. Neck
    - 在网络的颈部，采用的是：FPN+PAN结构，进行丰富的特征融合，这一部分和YOLOv4的结构相同。详细内容可参考：
        >https://blog.csdn.net/wjinjie/article/details/116793973
        >https://blog.csdn.net/wjinjie/article/details/107509243

4. Head
    - 对于网络的输出，遵循YOLO系列的一贯做法，采用的是耦合的Head。并且和YOLOv3、YOLOv4类似，采用了三个不同的输出Head，进行多尺度预测。详细内容可参考：
        >https://blog.csdn.net/wjinjie/article/details/116793973
        >https://blog.csdn.net/wjinjie/article/details/107509243

5. 改进方法
    - 自适应锚框计算
     在YOLOv3、YOLOv4中，是通过K-Means方法来获取数据集的最佳anchors，这部分操作需要在网络训练之前单独进行。为了省去这部分"额外"的操作，Yolov5的作者将此功能嵌入到整体代码中，每次训练时，自适应的计算不同训练集中的最佳锚框值。当然，如果觉得计算的锚框效果不是很好，也可以在代码中将自动计算锚框功能关闭。

    - 自适应灰度填充
    为了应对输入图片尺寸 不一的问题，为了避免这种情况的发生，YOLOv5采用了灰度填充的方式统一输入尺寸，避免了目标变形的问题。灰度填充的核心思想就是将原图的长宽等比缩放对应统一尺寸，然后对于空白部分用灰色填充。

### 4.6、模型总结
在目标检测领域里，YOLO、SSD的提出无疑具有里程碑的意义，其提供了端到端的检测框架，将目标检测成功应用到实时检测场景中，众多学者根据其核心思想发布了很多相对应的改进算法，例如基于SSD的Mobilenet-SSD则实现了手机终端上的实时目标检测，相比R-CNN系列：

**优点**:
- 快速，pipline简单
- 背景误检率低
- 通用性强

**缺点**：
- 识别物体位置精准性差
- 召回率低



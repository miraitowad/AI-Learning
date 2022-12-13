import torchvision.datasets as dsets
import torchvision.transforms as transforms



def get_data(BATCH_SIZE=8):
    transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 即先随机采集，然后对裁剪得到的图像缩放为同一大小
    transforms.RandomHorizontalFlip(),  # 以给定的概率随机水平旋转给定的PIL的图像，默认为0.5
    transforms.ToTensor(),  # 转化为向量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # 归一化函数
    ])
    trainData = dsets.ImageFolder('VGG16/data/imagenet/train', transform) # 训练数据
    testData = dsets.ImageFolder('VGG16/data/imagenet/test', transform)

    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)  # 加载训练
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)  # 加载测试
    return trainLoader,testLoader

import torch
import torch.nn as tnn
from model import *
from dataload import get_data

# 超参数设定
BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCH = 50
N_CLASSES = 2

# 载入数据
trainLoader,testLoader = get_data(BATCH_SIZE)

# 实例化模型
vgg16 = vgg16(N_CLASSES)
vgg16.cuda()  # 使用GPU跑数据

# Loss损失函数, Optimizer优化器 & Scheduler调整器
cost = tnn.CrossEntropyLoss()  # 交叉熵函数
optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE) # Adam优化器，动态学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer) # 根据测试指标下降学习率

# 开始训练
for epoch in range(EPOCH):

    avg_loss = 0
    cnt = 0
    for images, labels in trainLoader:
        # 使用GPU
        images = images.cuda()
        labels = labels.cuda()
        # 梯度归零
        optimizer.zero_grad()
        # 前向传播
        outputs = vgg16(images)
        # 计算损失函数
        loss = cost(outputs, labels)
        avg_loss += loss.data
        cnt += 1
        print("[E: %d] loss: %f, avg_loss: %f" % (epoch, loss.data, avg_loss / cnt))
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()
    # 调整学习率
    scheduler.step(avg_loss)

# 测试模型
vgg16.eval()
correct = 0
total = 0

for images, labels in testLoader:
    images = images.cuda()
    _, outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
    print(predicted, labels, correct, total)
    print("avg acc: %f" % (100 * correct / total))

# Save the Trained Model
torch.save(vgg16.state_dict(), 'vgg16.pt')
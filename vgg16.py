#!/bin/usr/env python3
# -*- coding utf-8 -*-
import os

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from matplotlib import pyplot as plt
from torchvision.models import vgg16

# from torchvision.models import VGG

"""
VGG是2014发布的，在图像分类上的ImageNet比赛上为当时的亚军，冠军是GoogLeNet。
在VGG论文中，主要应用的模型是VGG16和VGG19，其中数字表示层数。
这里实现的是VGG16。
"""


# 创建模型
class My_VGG16(nn.Module):

    def __init__(self, num_classes=5, init_weight=True):
        super(My_VGG16, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 分类层
        self.classifier = nn.Sequential(
            # 全连接的第一层，输入肯定是卷积输出的拉平值，即6*6*256
            # 输出是由AlexNet决定的，为4096
            nn.Linear(in_features=7 * 7 * 512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 最后一层，输出1000个类别，也是我们所说的softmax层
            nn.Linear(in_features=4096, out_features=7)
        )

        # 参数初始化
        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # 不要忘记在卷积--全连接的过程中，需要将数据拉平，之所以从1开始拉平，是因为我们
        # 批量训练，传入的x为[batch（每批的个数）,x(长),x（宽）,x（通道数）]，因此拉平需要从第1（索引，相当于2）开始拉平
        # 变为[batch,x*x*x]
        x = torch.flatten(x, 1)
        result = self.classifier(x)
        return result


class My_Dataset(Dataset):

    def __init__(self, filename, transform=None):
        self.filename = filename
        self.transform = transform
        self.image_name, self.label_image = self.operate_file()

    def __len__(self):
        return len(self.image_name)

    def __getitem__(self, idx):
        image = Image.open(self.image_name[idx])
        trans = T.RandomResizedCrop(224)
        image = trans(image)
        label = self.label_image[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.array(label))
        return image, label

    def operate_file(self):
        dir_list = os.listdir(self.filename)

        full_path = [self.filename + "/" + name for name in dir_list]

        name_list = []

        for i, v in enumerate(full_path):
            temp = os.listdir(v)
            temp_list = [v + '/' + j for j in temp]
            name_list.extend(temp_list)

        label_list = []

        temp_list = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        for j in range(7):
            for i in range(20):
                label_list.append(temp_list[j])

        return name_list, label_list


class My_Dataset_test(My_Dataset):

    def operate_file(self):

        dir_list = os.listdir(self.filename)

        full_path = [self.filename + "/" + name for name in dir_list]

        name_list = []

        for i, v in enumerate(full_path):
            temp = os.listdir(v)
            temp_list = [v + '/' + j for j in temp]
            name_list.extend(temp_list)

        label_list = []

        temp_list = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        for j in range(7):
            for i in range(9):
                label_list.append(temp_list[j])

        return name_list, label_list



loss_save = []
flag = 0
lr = 0.002


def adjust_lx(loss):
    global flag, lr
    loss_save.append(loss)
    if len(loss_save) >= 2:
        if abs(loss_save[-1] - loss_save[-2] <= 0.0005):
            flag += 1
        if loss_save[-1] - loss_save[-2] >= 0:
            flag += 1
    if flag >= 3:

        lr /= 10
        print(f"学习率已改变，变为了{lr}")
        flag = 0


def load_pretrained():
    device = torch.device("cuda:0")
    path = 'emo12.pth'
    model = My_VGG16()
    model.load_state_dict(torch.load(path, map_location="cuda:0"))
    model.to(device)
    return model


def train():
    batch_size = 7
    model = load_pretrained()
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    device = torch.device("cuda:0")
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    train_set = My_Dataset('archive/jaffe/train', T.ToTensor())
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    for i in range(1000):
        loss_temp = 0
        for j, (batch_data, batch_label) in enumerate(train_loader):
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            optimizer.zero_grad()
            prediction = model(batch_data)
            loss = loss_func(prediction, batch_label)
            loss_temp += loss.item()
            loss.backward()
            optimizer.step()
        print('[%d] loss: %.4f' % (i + 1, loss_temp / len(train_loader)))
    torch.save(model.state_dict(), 'emo12.pth')
    test(model)

def test(model):
    model.eval()
    # 批量数目
    batch_size = 7
    # 预测正确个数
    correct = 0
    # 加载数据
    test_set = My_Dataset_test('archive/jaffe/test', transform=T.ToTensor())
    test_loader = DataLoader(test_set, batch_size, shuffle=False)
    # 开始
    for batch_data, batch_label in test_loader:
        # 放入GPU中
        batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
        # batch_data, batch_label = batch_data.cpu(), batch_label.cpu()
        # 预测
        prediction = model(batch_data)
        # 将预测值中最大的索引取出，其对应了不同类别值
        predicted = torch.max(prediction.data, 1)[1]
        # 获取准确个数
        correct += (predicted == batch_label).sum()
    print('准确率: %.2f %%' % (100 * correct / 63))  # 因为总共63个测试数据
    return 100 * correct / 63


if __name__ == '__main__':
    train()








# coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import copy
import torchvision
from torchvision import transforms as T
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils import data
# from data import VehicleID_MC, VehicleID_All, id2name
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.font_manager import *
from collections import defaultdict

from InitRepNet import InitRepNet


# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

# 解决负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']

def train_mc(freeze_feature,
             resume=None):
    """
    训练RepNet: RAModel and color multi-label classification
    :param freeze_feature:
    :return:
    """
    net = RepNet(out_ids=10086,
                 out_attribs=257).cuda()
    print('=> Mix difference network:\n', net)

    # 是否从断点启动
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))  # 加载模型
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 数据集
    train_set = VehicleID_MC(root='/mnt/diskb/even/VehicleID_V1.0',
                             transforms=None,
                             mode='train')
    test_set = VehicleID_MC(root='/mnt/diskb/even/VehicleID_V1.0',
                            transforms=None,
                            mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=32,
                                              shuffle=False,
                                              num_workers=2)

    # 损失函数
    loss_func = torch.nn.CrossEntropyLoss().cuda()

    # 优化函数
    if freeze_feature:  # 锁住特征提取层，仅打开FC层
        optimizer = torch.optim.SGD(net.branch_1_fc.parameters(),
                                    lr=1e-3,
                                    momentum=9e-1,
                                    weight_decay=1e-8)
        for param in net.branch_1_feats.parameters():
            param.requires_grad = False
        print('=> optimize only FC layers.')
    else:  # 打开所有参数
        optimizer = torch.optim.SGD(net.branch_1.parameters(),
                                    lr=1e-3,
                                    momentum=9e-1,
                                    weight_decay=1e-8)
        print('=> optimize all layers.')

    # 开始训练
    print('\nTraining...')
    net.train()  # train模式

    best_acc = 0.0
    best_epoch = 0

    print('=> Epoch\tTrain loss\tTrain acc\tTest acc')
    for epoch in range(50):
        epoch_loss = []
        num_correct = 0
        num_total = 0

        for data, label in train_loader:  # 遍历每一个batch
            # ------------- 放入GPU
            data, label = data.cuda(), label.cuda().long()

            # ------------- 清空梯度
            optimizer.zero_grad()

            # ------------- 前向计算
            output = net.forward(X=data, branch=1)

            # 计算loss
            loss_m = loss_func(output[:, :250], label[:, 0])
            loss_c = loss_func(output[:, 250:], label[:, 1])
            loss = loss_m + loss_c

            # ------------- 统计
            epoch_loss.append(loss.item())

            # 统计样本数量
            num_total += label.size(0)

            # 统计训练数据正确率
            pred = get_predict_mc(output)
            label = label.cpu().long()
            num_correct += count_correct(pred=pred, label=label)

            # ------------- 反向运算
            loss.backward()
            optimizer.step()

        # 计算训练集准确度
        train_acc = 100.0 * float(num_correct) / float(num_total)

        # 计算测试集准确度
        test_acc = test_mc_accuracy(net=net,
                                    data_loader=test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1

            # 保存模型权重
            model_save_name = 'epoch_' + str(epoch + 1) + '.pth'
            torch.save(net.state_dict(),
                       '/mnt/diskb/even/MDNet_ckpt_br1/' + model_save_name)
            print('<= {} saved.'.format(model_save_name))

        print('\t%d \t%4.3f \t\t%4.2f%% \t\t%4.2f%%' %
              (epoch + 1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
    print('=> Best accuracy at epoch %d, test accuaray %f' % (best_epoch, best_acc))

def test_mc_accuracy(net,
                     data_loader):
    """
    :param net:
    :param data_loader:
    :return:
    """
    net.eval()  # 测试模式,前向计算

    num_correct = 0
    num_total = 0

    # 每个属性的准确率
    num_model = 0
    num_color = 0
    total_time = 0.0

    print('=> testing...')
    for data, label in data_loader:
        # 放入GPU.
        data, label = data.cuda(), label.cuda()

        # 将label转化为cpu, long
        label = label.cpu().long()

        # 前向运算, 预测
        output = net.forward(X=data, branch=1)  # 默认在device(GPU)中推理运算
        pred = get_predict_mc(output)  # 返回的pred存在于host端

        # 统计总数
        num_total += label.size(0)

        # 统计全部属性都预测正确正确数
        num_correct += count_correct(pred, label)

        # 统计各属性正确率
        num_model += count_attrib_correct(pred, label, 0)
        num_color += count_attrib_correct(pred, label, 1)

    # 总统计
    accuracy = 100.0 * float(num_correct) / float(num_total)
    model_acc = 100.0 * float(num_model) / float(num_total)
    color_acc = 100.0 * float(num_color) / float(num_total)

    print('=> test accuracy: {:.3f}%, RAModel accuracy: {:.3f}%, '
          'color accuracy: {:.3f}%'.format(
        accuracy, model_acc, color_acc))
    return accuracy

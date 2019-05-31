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


# --------------------------------------
# VehicleID用于MDNet
class VehicleID_All(data.Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 mode='train'):
        """
        :param root:
        :param transforms:
        :param mode:
        """
        if not os.path.isdir(root):
            print('[Err]: invalid root.')
            return

        # 加载图像绝对路径和标签
        if mode == 'train':
            txt_f_path = root + '/attribute/train_all.txt'
        elif mode == 'test':
            txt_f_path = root + '/attribute/test_all.txt'

        if not os.path.isfile(txt_f_path):
            print('=> [Err]: invalid txt file.')
            return

        # 打开vid2TrainID和trainID2Vid映射
        vid2TrainID_path = root + '/attribute/vid2TrainID.pkl'
        trainID2Vid_path = root + '/attribute/trainID2Vid.pkl'
        if not (os.path.isfile(vid2TrainID_path) \
                and os.path.isfile(trainID2Vid_path)):
            print('=> [Err]: invalid vid, train_id mapping file path.')

        with open(vid2TrainID_path, 'rb') as fh_1, \
                open(trainID2Vid_path, 'rb') as fh_2:
            self.vid2TrainID = pickle.load(fh_1)
            self.trainID2Vid = pickle.load(fh_2)

        self.imgs_path, self.lables = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip().split()
                img_path = root + '/image/' + line[0] + '.jpg'
                if os.path.isfile(img_path):
                    self.imgs_path.append(img_path)

                    tr_id = self.vid2TrainID[int(line[3])]
                    label = np.array([int(line[1]),
                                      int(line[2]),
                                      int(tr_id)], dtype=int)
                    self.lables.append(torch.Tensor(label))

        assert len(self.imgs_path) == len(self.lables)
        print('=> total %d samples loaded in %s mode' % (len(self.imgs_path), mode))

        # 加载数据变换
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        """
        关于数据缩放方式: 先默认使用非等比缩放
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # 数据变换, 灰度图转换成'RGB'
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # 图像数据变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.lables[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)


# Vehicle ID用于车型和颜色的多标签分类
class VehicleID_MC(data.Dataset):
    def __init__(self,
                 root,
                 transforms=None,
                 mode='train'):
        """
        :param root:
        :param transforms:
        :param mode:
        """
        if not os.path.isdir(root):
            print('[Err]: invalid root.')
            return

        # 加载图像绝对路径和标签
        if mode == 'train':
            txt_f_path = root + '/attribute/train.txt'
        elif mode == 'test':
            txt_f_path = root + '/attribute/test.txt'

        if not os.path.isfile(txt_f_path):
            print('=> [Err]: invalid txt file.')
            return

        self.imgs_path, self.lables = [], []
        with open(txt_f_path, 'r', encoding='utf-8') as f_h:
            for line in f_h.readlines():
                line = line.strip().split()
                img_path = root + '/image/' + line[0] + '.jpg'
                if os.path.isfile(img_path):
                    self.imgs_path.append(img_path)
                    label = np.array([int(line[1]), int(line[2])], dtype=int)
                    self.lables.append(torch.Tensor(label))

        assert len(self.imgs_path) == len(self.lables)
        print('=> total %d samples loaded in %s mode' % (len(self.imgs_path), mode))

        # 加载数据变换
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        """
        关于数据缩放方式: 先默认使用非等比缩放
        :param idx:
        :return:
        """
        img = Image.open(self.imgs_path[idx])

        # 数据变换, 灰度图转换成'RGB'
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # 图像数据变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, self.lables[idx]

    def __len__(self):
        """
        :return:
        """
        return len(self.imgs_path)

# --------------------------------------- methods
def get_predict_mc(output):
    """
    softmax归一化,然后统计每一个标签最大值索引
    :param output:
    :return:
    """
    # 计算预测值
    output = output.cpu()  # 从GPU拷贝出来
    pred_model = output[:, :250]
    pred_color = output[:, 250:]

    model_idx = pred_model.max(1, keepdim=True)[1]
    color_idx = pred_color.max(1, keepdim=True)[1]

    # 连接pred
    pred = torch.cat((model_idx, color_idx), dim=1)
    return pred


def count_correct(pred, label):
    """
    :param output:
    :param label:
    :return:
    """
    assert pred.size(0) == label.size(0)
    correct_num = 0
    for one, two in zip(pred, label):
        if torch.equal(one, two):
            correct_num += 1
    return correct_num


def count_attrib_correct(pred, label, idx):
    """
    :param pred:
    :param label:
    :param idx:
    :return:
    """
    assert pred.size(0) == label.size(0)
    correct_num = 0
    for one, two in zip(pred, label):
        if one[idx] == two[idx]:
            correct_num += 1
    return correct_num




def gen_test_pairs(test_txt,
                   dst_dir,
                   num=10000):
    """
    生成测试pair数据: 一半positive，一半negative
    :param test_txt:
    :return:
    """
    if not os.path.isfile(test_txt):
        print('[Err]: invalid file.')
        return
    print('=> genarating %d samples...' % num)

    with open(test_txt, 'r') as f_h:
        valid_list = f_h.readlines()
        print('=> %s loaded.' % test_txt)

        # 映射: img_name => cls_id
        valid_dict = {x.strip().split()[0]: int(x.strip().split()[3]) for x in valid_list}

        # 映射: cls_id => img_list
        inv_dict = defaultdict(list)
        for k, v in valid_dict.items():
            inv_dict[v].append(k)

        # 统计样本数不少于2的id
        big_ids = [k for k, v in inv_dict.items() if len(v) > 1]

    # 添加测试样本
    pair_set = set()
    while len(pair_set) < num:
        if random.random() <= 0.7:  # positive
            # 随机从big_ids中选择一个
            pick_id = random.sample(big_ids, 1)[0]  # 不放回抽取

            anchor = random.sample(inv_dict[pick_id], 1)[0]
            positive = random.choice(inv_dict[pick_id])
            while positive == anchor:
                positive = random.choice(inv_dict[pick_id])

            pair_set.add(anchor + '\t' + positive + '\t1')
        else:  # negative
            pick_id_1 = random.sample(big_ids, 1)[0]  # 不放回抽取
            pick_id_2 = random.sample(big_ids, 1)[0]  # 不放回抽取
            while pick_id_2 == pick_id_1:
                pick_id_2 = random.sample(big_ids, 1)[0]
            assert pick_id_2 != pick_id_1
            anchor = random.choice(inv_dict[pick_id_1])
            negative = random.choice(inv_dict[pick_id_2])

            pair_set.add(anchor + '\t' + negative + '\t0')
    print(list(pair_set)[:5])
    print(len(pair_set))

    # 序列化pair_set到dst_dir
    pair_set_f_path = dst_dir + '/' + 'pair_set_vehicle.txt'
    with open(pair_set_f_path, 'w') as f_h:
        for x in pair_set:
            f_h.write(x + '\n')
    print('=> %s generated.' % pair_set_f_path)


# 获取每张测试图片对应的特征向量
def gen_feature_map(resume,
                    imgs_path,
                    batch_size=16):
    """
    根据图相对生成每张图象的特征向量, 映射: img_name => img_feature vector
    :param resume:
    :param imgs_path:
    :return:
    """
    net = nn.DataParallel(RepNet(out_ids=10086,out_attribs=257)).cuda()
    print('=> Mix difference network:\n', net)

    # 从断点启动
    if resume is not None:
        if os.path.isfile(resume):
            # 加载模型
            net.load_state_dict(torch.load(resume))
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 图像数据变换
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    # load model, image and forward
    data, features = None, None
    for i, img_path in tqdm(enumerate(imgs_path)):
        # load image
        img = Image.open(img_path)

        # tuen to RGB
        if img.mode == 'L' or img.mode == 'I':  # 8bit或32bit灰度图
            img = img.convert('RGB')

        # image data transformations
        img = transforms(img)
        img = img.view(1, 3, 224, 224)

        if data is None:
            data = img
        else:
            data = torch.cat((data, img), dim=0)

        if data.shape[0] % batch_size == 0 or i == len(imgs_path) - 1:

            # collect a batch of image data
            data = data.cuda()

            output = net.forward(X=data,
                                 branch=5,
                                 label=None)

            batch_features = output.data.cpu().numpy()
            if features is None:
                features = batch_features
            else:
                features = np.vstack((features, batch_features))

            # clear a batch of images
            data = None

    # generate feature map
    feature_map = {}
    for i, img_path in enumerate(imgs_path):
        feature_map[img_path] = features[i]

    print('=> feature map size: %d' % (len(feature_map)))
    return feature_map


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    """
    :param y_score:
    :param y_true:
    :return:
    """
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        print('=> th: %.3f, acc: %.3f' % (th, acc))

        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


# 统计阈值和准确率: Vehicle ID数据集
def get_th_acc_VID(resume,
                   pair_set_txt,
                   img_dir,
                   batch_size=16):
    """
    :param resume:
    :param pair_set_txt:
    :param img_dir:
    :param batch_size:
    :return:
    """
    if not os.path.isfile(pair_set_txt):
        print('=> [Err]: invalid file.')
        return

    pairs, imgs_path = [], []
    with open(pair_set_txt, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            pair = line.strip().split()

            imgs_path.append(img_dir + '/' + pair[0] + '.jpg')
            imgs_path.append(img_dir + '/' + pair[1] + '.jpg')

            pairs.append(pair)

    print('=> total %d pairs.' % (len(pairs)))
    print('=> total %d image samples.' % (len(imgs_path)))
    imgs_path.sort()

    # generate feature dict
    feature_map = gen_feature_map(resume=resume,
                                  imgs_path=imgs_path,
                                  batch_size=batch_size)

    sims, labels = [], []
    for pair in pairs:
        img_path_1 = img_dir + '/' + pair[0] + '.jpg'
        img_path_2 = img_dir + '/' + pair[1] + '.jpg'
        sim = cosin_metric(feature_map[img_path_1],
                           feature_map[img_path_2])
        label = int(pair[2])
        sims.append(sim)
        labels.append(label)

    # 统计最佳阈值及其对应的准确率
    acc, th = cal_accuracy(sims, labels)
    print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
    return acc, th


# 统计阈值和准确率: Car Match数据集
def test_car_match_data(resume,
                        pair_set_txt,
                        img_root,
                        batch_size=16):
    """
    :param resume:
    :param pair_set_txt:
    :param batch_size:
    :return:
    """
    if not os.path.isfile(pair_set_txt):
        print('=> [Err]: invalid file.')
        return

    pairs, imgs_path = [], []
    with open(pair_set_txt, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = line.strip().split()

            imgs_path.append(img_root + '/' + line[0])
            imgs_path.append(img_root + '/' + line[1])

            pairs.append(line)

    print('=> total %d pairs.' % (len(pairs)))
    print('=> total %d image samples.' % (len(imgs_path)))
    imgs_path.sort()

    # 计算特征向量字典
    feature_map = gen_feature_map(resume=resume,
                                  imgs_path=imgs_path,
                                  batch_size=batch_size)

    # 计算所有pair的sim
    sims, labels = [], []
    for pair in pairs:
        img_path_1 = img_root + '/' + pair[0]
        img_path_2 = img_root + '/' + pair[1]
        sim = cosin_metric(feature_map[img_path_1],
                           feature_map[img_path_2])
        label = int(pair[2])
        sims.append(sim)
        labels.append(label)

    # 统计最佳阈值及其对应的准确率
    acc, th = cal_accuracy(sims, labels)
    print('=> best threshold: %.3f, accuracy: %.3f%%' % (th, acc * 100.0))
    return acc, th


def test_accuracy(net, data_loader):
    """
    测试VehicleID分类在测试集上的准确率
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
        data, label = data.cuda(), label.cuda().long()

        # 前向运算, 预测Vehicle ID
        output = net.forward(X=data,
                             branch=3,
                             label=label[:, 2])

        # 统计总数
        num_total += label.size(0)

        # 统计全部属性都预测正确正确数
        _, pred = torch.max(output.data, 1)
        batch_correct = (pred == label[:, 2]).sum().item()
        num_correct += batch_correct

    # test-set总的统计
    accuracy = 100.0 * float(num_correct) / float(num_total)
    print('=> test accuracy: {:.3f}%'.format(accuracy))

    return accuracy




if __name__ == '__main__':
    # test_init_weight()

    # train_mc(freeze_feature=False,
    #          resume='/mnt/diskb/even/MDNet_ckpt_br1/epoch_16.pth')

    # train(resume='/mnt/diskb/even/MDNet_ckpt_br1/epoch_16.pth')
    train(resume=None)  # 从头开始训练

    # train(resume='/mnt/diskb/even/MDNet_ckpt_all/epoch_24_bk.pth')

    # -----------------------------------
    # viz_results(resume='/mnt/diskb/even/MDNet_ckpt_all/epoch_12.pth',
    #             data_root='/mnt/diskb/even/VehicleID_V1.0')

    # test_car_match_data(resume='/mnt/diskb/even/MDNet_ckpt_all/epoch_10.pth',
    #                     pair_set_txt='/mnt/diskc/even/Car_DR/ArcFace_pytorch/data/pair_set_car.txt',
    #                     img_root='/mnt/diskc/even/CarReIDCrop',  # CarReID_data
    #                     batch_size=16)

    # get_th_acc_VID(resume='/mnt/diskb/even/MDNet_ckpt_all/epoch_10.pth',
    #                pair_set_txt='/mnt/diskb/even/VehicleID_V1.0/attribute/pair_set_vehicle.txt',
    #                img_dir='/mnt/diskb/even/VehicleID_V1.0/image',
    #                batch_size=16)

    print('=> Done.')

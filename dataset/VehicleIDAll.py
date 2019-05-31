from torch.utils import data
import os
import pickle
from torchvision import transforms as T
import numpy as np
import torch
from PIL import Image

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

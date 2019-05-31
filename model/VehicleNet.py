import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    #CBAM: Convolutional Block Attention Module, ECCV2018
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class VehicleReid(nn.Module):
    def __init__(self):
        """
        ResNet model 
        """
        super(VehicleReid, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-3])
        self.base1 = nn.Sequential(*list(resnet50.children())[-3:-1])
        self.SpatialAttention = SpatialAttention()
        self.mfc = nn.Linear(2048, 250) # VehicleID 数据集250类种类
        self.cfc = nn.Linear(2048,7) # VehicleID 数据集7类颜色
        self.ifc = nn.Linear(2048, 10086)  # VehicleID 数据集10086个ID

    def forward(self, x, only_feature):
        x = self.base(x)
        x = self.SpatialAttention(x) * x
        x = self.base1(x)
        feature = x.view(x.size(0), -1)

        mid = self.mfc(feature)
        cid = self.cfc(feature)
        id = self.ifc(feature)

        if only_feature:
            return feature

        return feature, mid, cid, id

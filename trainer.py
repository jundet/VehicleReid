# coding=utf-8

import torch
import torchvision
import argparse
from torch.utils import data
from dataset.VehicleIDAll import VehicleID_All
from utils.loss import FocalLoss
import matplotlib as mpl
from matplotlib.font_manager import *
from model.reidnet import InitRepNet
from utils.eval import test_accuracy

# 解决负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']

parser = argparse.ArgumentParser(description='VehicleReid.')
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--batch', default=32, type=int, help="batch size")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

def train(resume):
    """
    :param resume:
    :return:
    """
    # net = RepNet(out_ids=10086,
    #              out_attribs=257).cuda()

    vgg16_pretrain = torchvision.models.vgg16(pretrained=True)
    net = torch.nn.DataParallel(InitRepNet(vgg_orig=vgg16_pretrain,
                     out_ids=10086,
                     out_attribs=257)).cuda()

    print('=> Mix difference network:\n', net)

    # whether to resume from checkpoint
    if resume is not None:
        if os.path.isfile(resume):
            net.load_state_dict(torch.load(resume))  # 加载模型
            print('=> net resume from {}'.format(resume))
        else:
            print('=> [Err]: invalid resume path @ %s' % resume)

    # 数据集
    train_set = VehicleID_All(root='/media/ml/F/de/dataset/VehicleID_V1.0/',
                              transforms=None,
                              mode='train')
    test_set = VehicleID_All(root='/media/ml/F/de/dataset/VehicleID_V1.0/',
                             transforms=None,
                             mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=args.batch,
                                              shuffle=False,
                                              num_workers=4)

    # loss function
    loss_func_1 = torch.nn.CrossEntropyLoss().cuda()
    loss_func_2 = FocalLoss(gamma=2).cuda()

    # optimization function
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=1e-3,
                                momentum=9e-1,
                                weight_decay=1e-8)
    print('=> optimize all layers.')

    # start to train
    print('\nTraining...')
    net.train()  # train模式

    best_acc = 0.0
    best_epoch = 0

    print('=> Epoch\tTrain loss\tTrain acc\tTest acc')
    for epoch_i in range(30):

        epoch_loss = []
        num_correct = 0
        num_total = 0
        for batch_i, (data, label) in enumerate(train_loader):  # 遍历每一个batch
            # ------------- put data to device
            data, label = data.cuda(), label.cuda().long()

            # ------------- clear gradients
            optimizer.zero_grad()

            # ------------- forward pass of 3 branches
            output_1 = net.forward(X=data, branch=1, label=None)
            output_2 = net.forward(X=data, branch=2, label=label[:, 2])
            output_3 = net.forward(X=data, branch=3, label=label[:, 2])

            # ------------- calculate loss
            # branch1 loss
            loss_m = loss_func_1(output_1[:, :250], label[:, 0])  # vehicle model
            loss_c = loss_func_1(output_1[:, 250:], label[:, 1])  # vehicle color
            loss_br1 = loss_m + loss_c

            # branch2 loss
            loss_br2 = loss_func_2(output_2, label[:, 2])

            # branch3 loss: Vehicle ID classification
            loss_br3 = loss_func_2(output_3, label[:, 2])

            # 加权计算总loss
            loss = 0.5 * loss_br1 + 0.5 * loss_br2 + 1.0 * loss_br3

            # ------------- statistics
            epoch_loss.append(loss.cpu().item())

            # count samples
            num_total += label.size(0)

            # statistics of correct number
            _, pred = torch.max(output_3.data, 1)
            batch_correct = (pred == label[:, 2]).sum().item()
            batch_acc = float(batch_correct) / float(label.size(0))
            num_correct += batch_correct

            # ------------- back propagation
            loss.backward()
            optimizer.step()

            iter_count = epoch_i * len(train_loader) + batch_i

            # output batch accuracy
            if iter_count % 10 == 0:
                print('=> epoch {} iter {:>4d}/{:>4d}'
                      ', total_iter {:>6d} '
                      '| loss {:>5.3f} | accuracy {:>.3%}'
                      .format(epoch_i + 1,
                              batch_i,
                              len(train_loader),
                              iter_count,
                              loss.item(),
                              batch_acc))

        # total epoch accuracy
        train_acc = float(num_correct) / float(num_total)
        print('=> epoch {} | average loss: {:.3f} | average accuracy: {:>.3%}'
              .format(epoch_i + 1,
                      float(sum(epoch_loss)) / float(len(epoch_loss)),
                      train_acc))

        # calculate test-set accuracy
        test_acc = test_accuracy(net=net,
                                 data_loader=test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch_i + 1

            # save model weights
            model_save_name = 'epoch_' + str(epoch_i + 1) + '.pth'
            torch.save(net.state_dict(),
                       './ckpt/' + model_save_name)
            print('<= {} saved.'.format(model_save_name))

        print('\t%d \t%4.3f \t\t%4.2f%% \t\t%4.2f%%' %
              (epoch_i + 1,
               sum(epoch_loss) / len(epoch_loss),
               train_acc * 100.0,
               test_acc))
    print('=> Best accuracy at epoch %d, test accuaray %f' % (best_epoch, best_acc))

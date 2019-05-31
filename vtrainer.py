# coding=utf-8

import torch
import torchvision
import argparse
from torch.utils import data
from torch.optim import lr_scheduler
from dataset.VehicleIDAll import VehicleID_All
import time
import datetime
import matplotlib as mpl
from matplotlib.font_manager import *
from utils.eval import test_accuracy
from model import VehicleNet
from utils.triplet import TripletLoss

# 解决负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParam4s['font.sans-serif'] = ['SimHei']

parser = argparse.ArgumentParser(description='VehicleReid.')
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--train-batch', default=64, type=int, help="batch size")
parser.add_argument('--test-batch', default=32, type=int, help="batch size")
parser.add_argument('--stepsize', default=100, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help="initial learning rate")
parser.add_argument('--max-epoch', default=400, type=int,
                    help="maximum epochs to run")
parser.add_argument('--eval-step', type=int, default=50,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
args = parser.parse_args()


def main():
    """
    :param resume:
    :return:
    """
    torch.manual_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    model = VehicleNet.VehicleReid()


    # whether to resume from checkpoint
    # if resume is not None:
    #     if os.path.isfile(resume):
    #         model.load_state_dict(torch.load(resume))  # 加载模型
    #         print('=> net resume from {}'.format(resume))
    #     else:
    #         print('=> [Err]: invalid resume path @ %s' % resume)

    # 数据集
    train_set = VehicleID_All(root='/media/ml/F/de/dataset/VehicleID_V1.0/',
                              transforms=None,
                              mode='train')
    test_set = VehicleID_All(root='/media/ml/F/de/dataset/VehicleID_V1.0/',
                             transforms=None,
                             mode='test')
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=args.train_batch,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=args.test_batch,
                                              shuffle=False,
                                              num_workers=4)

    model = torch.nn.DataParallel(model).cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    # loss function
    criterion_xent = torch.nn.CrossEntropyLoss()
    criterion_htri = TripletLoss(margin=0.3)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()
    print("==> Start training")
    for epoch in range(args.max_epoch):
        train(train_loader, optimizer, model, epoch, criterion_xent, criterion_htri)
        if args.stepsize > 0: scheduler.step()

        # calculate test-set accuracy
        if args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch:
            print("==> Start Test")
            test_acc = test_accuracy(model, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1

                # save model weights
                model_save_name = 'epoch_' + str(epoch + 1) + '.pth'
                torch.save(model.state_dict(),
                           './ckpt/' + model_save_name)
                print('<= {} saved.'.format(model_save_name))

    print('=> Best accuracy at epoch %d, test accuaray %f' % (best_epoch, best_acc))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(train_loader, optimizer, model,epoch, criterion_xent, criterion_htri):
    model.train()
    for batch_i, (data, label) in enumerate(train_loader):
        data, label = data.cuda(), label.cuda().long()
        optimizer.zero_grad()
        feature, mid, cid, id = model(data, False)
        xent_loss = criterion_xent(id, label[:, 2])  # ID
        htri_loss = criterion_htri(feature, label[:, 2])  # ID

        xent_loss += criterion_xent(mid, label[:, 0])  # model
        xent_loss += criterion_xent(cid, label[:, 1])  # color
        loss = xent_loss + htri_loss
        loss.backward()
        optimizer.step()

        iter_count = epoch * len(train_loader) + batch_i

        if iter_count % 100 == 0:
            print('=> epoch {} iter {:>4d}/{:>4d}'
                  ', total_iter {:>6d} '
                  '| loss {:>5.3f}'
                  .format(epoch + 1,
                          batch_i,
                          len(train_loader),
                          iter_count,
                          loss.item()))


if __name__ == '__main__':
    main()

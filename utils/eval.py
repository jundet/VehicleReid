import torch

def test_accuracy(model, data_loader):
    """
    测试VehicleID分类在测试集上的准确率
    :param net:
    :param data_loader:
    :return:
    """
    model.eval()

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

        feature, mid, cid, id = model(data,False)
        # 统计总数
        num_total += label.size(0)

        # 统计全部属性都预测正确正确数
        _, pred = torch.max(id, 1)
        batch_correct = (pred == label[:, 2]).sum().item()
        num_correct += batch_correct

    # test-set总的统计
    accuracy = 100.0 * float(num_correct) / float(num_total)
    print('=> test accuracy: {:.3f}%'.format(accuracy))

    return accuracy

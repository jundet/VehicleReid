
# @TODO: 可视化分类结果...
def ivt_tensor_img(input,
                   title=None):
    """
    Imshow for Tensor.
    """
    input = input.numpy().transpose((1, 2, 0))

    # 转变数组格式 RGB图像格式：rows * cols * channels
    # 灰度图则不需要转换，只有(rows, cols)而不是（rows, cols, 1）
    # (3, 228, 906)   #  (228, 906, 3)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 去标准化，对应transforms
    input = std * input + mean

    # 修正 clip 限制inp的值，小于0则=0，大于1则=1
    output = np.clip(input, 0, 1)

    # plt.imshow(input)
    # if title is not None:
    #     plt.title(title)
    # plt.pause(0.001)  # pause a bit so that plots are updated

    return output


def viz_results(resume,
                data_root):
    """
    :param resume:
    :param data_root:
    :return:
    """
    color_dict = {'black': u'黑色',
                  'blue': u'蓝色',
                  'gray': u'灰色',
                  'red': u'红色',
                  'sliver': u'银色',
                  'white': u'白色',
                  'yellow': u'黄色'}

    test_set = VehicleID_All(root=data_root,
                             transforms=None,
                             mode='test')
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1)

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

    # 测试模式
    net.eval()

    # 加载类别id映射和类别名称
    modelID2name_path = data_root + '/attribute/modelID2name.pkl'
    colorID2name_path = data_root + '/attribute/colorID2name.pkl'
    trainID2Vid_path = data_root + '/attribute/trainID2Vid.pkl'
    if not (os.path.isfile(modelID2name_path) and \
            os.path.isfile(colorID2name_path) and \
            os.path.isfile((trainID2Vid_path))):
        print('=> [Err]: invalid file.')
        return

    with open(modelID2name_path, 'rb') as fh_1, \
            open(colorID2name_path, 'rb') as fh_2, \
            open(trainID2Vid_path, 'rb') as fh_3:
        modelID2name = pickle.load(fh_1)
        colorID2name = pickle.load(fh_2)
        trainID2Vid = pickle.load(fh_3)

    # 测试
    print('=> testing...')
    for i, (data, label) in enumerate(test_loader):
        # 放入GPU.
        data, label = data.cuda(), label.cuda().long()

        # 前向运算: 预测车型、车身颜色
        output_attrib = net.forward(X=data,
                                    branch=1,
                                    label=None)
        pred_mc = get_predict_mc(output_attrib).cpu()[0]
        pred_m_id, pred_c_id = pred_mc[0].item(), pred_mc[1].item()
        pred_m_name = modelID2name[pred_m_id]
        pred_c_name = colorID2name[pred_c_id]

        # 前向运算: 预测Vehicle ID
        output_id = net.forward(X=data,
                                branch=3,
                                label=label[:, 2])
        _, pred_tid = torch.max(output_id, 1)
        pred_tid = pred_tid.cpu()[0].item()
        pred_vid = trainID2Vid[pred_tid]

        # 获取实际result
        img_path = test_loader.dataset.imgs_path[i]
        img_name = os.path.split(img_path)[-1][:-4]

        result = label.cpu()[0]
        res_m_id, res_c_id, res_vid = result[0].item(), result[1].item(), \
                                      trainID2Vid[result[2].item()]
        res_m_name = modelID2name[res_m_id]
        res_c_name = colorID2name[res_c_id]

        # 图像标题
        title = 'pred: ' + pred_m_name + ' ' + color_dict[pred_c_name] \
                + ', vehicle ID ' + str(pred_vid) \
                + '\n' + 'resu: ' + res_m_name + ' ' + color_dict[res_c_name] \
                + ', vehicle ID ' + str(res_vid)
        print('=> result: ', title)

        # 绘图
        img = ivt_tensor_img(data.cpu()[0])
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(title)
        plt.show()
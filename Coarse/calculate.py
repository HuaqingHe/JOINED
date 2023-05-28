import os


def main():

    # dice = 0
    # dices = []
    # jaccard = 0
    # acc = 0
    # kappa = 0
    # # _size = 'ori'
    # # _size = 'ori_crop192'
    # # _size = 'aug'
    # _size = 'aug_crop192'
    # # _type = 'unet++'
    # _type = 'unet++'
    # encode = 'vgg16_bn'
    # # encode = 'resnet50'
    # # encode = 'resnext50_32x4d'
    # # encode = 'timm-resnest50d'

    # print(_type, encode, _size)

    # for i in range(1, 6):
    #     base_path = './results/{}/{}/{}_imagenet_dice_{}'.format(i, _size, _type, encode)
    #     d_max = 0
    #     for file in os.listdir(base_path):
    #         if os.path.isfile(os.path.join(base_path, file)):
    #             indexs = file.split('_')

    #             if float(indexs[2]) > d_max:
    #                 d = float(indexs[2])
    #                 j = float(indexs[4])
    #                 a = float(indexs[6])
    #                 k = float(indexs[8].split('.pt')[0])
    #                 d_max = d

    #     dice += d
    #     dices.append(d)
    #     jaccard += j
    #     acc += a
    #     kappa += k

    # dice = round(dice/5, 4)
    # jaccard = round(jaccard/5, 4)
    # acc = round(acc/5, 4)
    # kappa = round(kappa/5, 4)

    # print('Best Dice: Dice:{}, Jaccard:{}, Acc:{}, Kappa:{}'.format(dice, jaccard, acc, kappa))

    # dice = 0
    # dices = []
    # jaccard = 0
    # acc = 0
    # kappa = 0

    # for i in range(1, 6):
    #     base_path = './results/{}/{}/{}_imagenet_dice_{}'.format(i, _size, _type, encode)
    #     d_max = 0
    #     for file in os.listdir(base_path):
    #         if os.path.isfile(os.path.join(base_path, file)):
    #             indexs = file.split('_')

    #             if float(indexs[6]) > d_max:
    #                 d = float(indexs[2])
    #                 j = float(indexs[4])
    #                 a = float(indexs[6])
    #                 k = float(indexs[8].split('.pt')[0])
    #                 d_max = a

    #     dice += d
    #     dices.append(d)
    #     jaccard += j
    #     acc += a
    #     kappa += k

    # dice = round(dice/5, 4)
    # jaccard = round(jaccard/5, 4)
    # acc = round(acc/5, 4)
    # kappa = round(kappa/5, 4)

    # print('Best Acc: Dice:{}, Jaccard:{}, Acc:{}, Kappa:{}'.format(dice, jaccard, acc, kappa))

    # dice = 0
    # dices = []
    # jaccard = 0
    # _size = 'ori_crop192'
    # _type = 'deeplabv3+'
    # encode = 'vgg11_bn'

    # print(_type, encode, _size)

    # for i in range(1, 6):
    #     base_path = './results/{}/seg/{}/{}_imagenet_dice_{}'.format(i, _size, _type, encode)
    #     d_max = 0
    #     for file in os.listdir(base_path):
    #         if os.path.isfile(os.path.join(base_path, file)):
    #             indexs = file.split('_')

    #             if float(indexs[2]) > d_max:
    #                 d = float(indexs[2])
    #                 j = float(indexs[4].split('.pt')[0])
    #                 d_max = d

    #     dice += d
    #     dices.append(d)
    #     jaccard += j
    # dice = round(dice/5, 4)
    # jaccard = round(jaccard/5, 4)

    # print('Best Dice: Dice:{}, Jaccard:{}'.format(dice, jaccard))

    acc = 0
    kappa = 0
    # _size = 'ori'
    _size = 'aug'
    # _type = 'unet++'
    _type = 'vgg16'
    encode = 'vgg16'
    # encode = 'resnet50'
    # encode = 'resnext50_32x4d'
    # encode = 'timm-resnest50d'
    # pretrain = 'True'
    pretrain = 'False'

    print(_type, _size, pretrain)

    for i in range(1, 6):
        base_path = './results3/{}/cls/{}/{}_{}_dice_{}'.format(i, _size, _type, pretrain, encode)
        a_max = 0
        for file in os.listdir(base_path):
            if os.path.isfile(os.path.join(base_path, file)):
                indexs = file.split('_')

                if float(indexs[2]) > a_max:
                    a = float(indexs[2])
                    k = float(indexs[4].split('.pt')[0])
                    a_max = a

        acc += a
        kappa += k

    acc = round(acc/5, 4)
    kappa = round(kappa/5, 4)

    print('Acc:{}, Kappa:{}'.format(acc, kappa))


if __name__ == "__main__":
    main()

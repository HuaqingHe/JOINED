import torch
import numpy as np
import torchvision
import time
import argparse
import segmentation_models_pytorch as smp
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from cv2 import cv2
from losses import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def mean_and_std(paths, num_list):
    print('Calculating mean and std of training set for data normalization.')
    m_list, s_list = [], []
    for i in tqdm(num_list):
        img_filename = paths + str(i) + '.bmp'
        img = cv2.imread(img_filename)
        (m, s) = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    m = m[0][::-1][0]/255
    s = s[0][::-1][0]/255

    return m, s


def build_model(model_type, encoder, pretrain, classnum, depth, activation, in_channel=3, aux=False):

    aux_params = dict(
        pooling='avg',             # one of 'avg', 'max'
        dropout=0.5,               # dropout ratio, default is None
        activation='sigmoid',      # activation function, default is None
        classes=3,                 # define number of output labels
    )

    if depth == 5:
        decoder_channels = (256, 128, 64, 32, 16)
    elif depth == 4:
        decoder_channels = (256, 128, 64, 32)
    elif depth == 3:
        decoder_channels = (256, 128, 64)

    if activation == 'None':
        activation = None

    if pretrain == 'None':
        pretrain = None

    if model_type == "unet_smp":
        model = smp.Unet(
            encoder_name=encoder,
            encoder_depth=depth,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            decoder_channels=decoder_channels,                
            decoder_attention_type=None,
            in_channels=in_channel,
            classes=classnum,
            activation=activation,
            aux_params=None  # if not aux else aux_params
        )
    if model_type == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_depth=depth,
            encoder_weights=pretrain,
            decoder_use_batchnorm=True,
            decoder_channels=decoder_channels,
            decoder_attention_type=None,
            in_channels=in_channel,
            classes=classnum,
            activation=activation,
            aux_params=None if not aux else aux_params
        )
    if model_type == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=pretrain,
            encoder_depth=3,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            in_channels=in_channel,
            classes=classnum,
            activation=activation,
            upsampling=8,
            aux_params=None if not aux else aux_params
        )
    if model_type == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights=pretrain,
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=in_channel,
            classes=classnum,
            activation=activation,
            upsampling=4,
            aux_params=None if not aux else aux_params
        )
    return model


def define_loss(loss_type, classnum, weights=[3, 1, 2]):

    if loss_type == "jaccard":
        if classnum == 3:
            criterion = train_JaccardLoss(mode='multiclass')
        elif classnum == 1:
            criterion = train_JaccardLoss(mode='binary')
        else:
            criterion = train_JaccardLoss(mode='multiclass')
    if loss_type == "dice":
        if classnum == 3:
            criterion = train_DiceLoss(mode='multiclass')
        elif classnum == 1:
            criterion = train_DiceLoss(mode='binary')
        else:
            criterion = train_DiceLoss(mode='multiclass')
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    if loss_type == 'bce':
        criterion = nn.BCELoss()
    if loss_type == "dicebce":
        criterion = DiceBCELoss()
    if loss_type == "bcewithlogit":
        criterion = smp.utils.losses.BCEWithLogitsLoss()

    return criterion


def weighted_define_loss(loss_type, classnum, weights=[3, 1, 2]):

    if loss_type == "jaccard":
        if classnum == 3:
            criterion = train_JaccardLoss_weighted(mode='multiclass')
        elif classnum == 1:
            criterion = train_JaccardLoss_weighted(mode='binary')
        else:
            criterion = train_JaccardLoss_weighted(mode='multiclass')
    if loss_type == "dice":
        if classnum == 3:
            criterion = train_DiceLoss_weighted(mode='multiclass')
        elif classnum == 1:
            criterion = train_DiceLoss_weighted(mode='binary')
        else:
            criterion = train_DiceLoss_weighted(mode='multiclass')
    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    if loss_type == "dicebce":
        criterion = DiceBCELoss()
    if loss_type == "bcewithlogit":
        criterion = smp.utils.losses.BCEWithLogitsLoss()

    return criterion


def create_train_arg_parser():

    parser = argparse.ArgumentParser(
        description="train setup for segmentation")
    parser.add_argument("--data_path", type=str, help="path to img jpg files")
    parser.add_argument("--mask_path", type=str, help="path to img jpg files")
    parser.add_argument("--val_data_path", type=str,
                        help="path to img jpg files")
    parser.add_argument("--val_mask_path", type=str,
                        help="path to img jpg files")
    parser.add_argument("--batch_size", type=int,
                        default=64, help="train batch size")
    parser.add_argument("--num_epochs", type=int,
                        default=500, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument(
        "--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="If use_pretrained is true, provide checkpoint.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: unet,dcan,dmtn,psinet,convmcd",
    )
    parser.add_argument("--LR_seg", default=1e-4,
                        type=float, help='learning rate.')
    parser.add_argument("--in_channels", default=3,
                        type=int, help='input channels.')
    parser.add_argument("--flod", default=1,
                        type=int, help='input flod.')
    parser.add_argument("--target", type=str, help="Model save target.")
    parser.add_argument("--save_path", type=str, help="Model save path.")
    parser.add_argument("--test_data_path", type=str, help="test data path.")
    parser.add_argument("--test_mask_path", type=str, help="test mask path.")
    parser.add_argument("--train_data_path", type=str, help="train data path.")
    parser.add_argument("--train_mask_path", type=str, help="train mask path.")
    parser.add_argument("--encoder", type=str, default=None, help="encoder.")
    parser.add_argument("--encoder_depth", type=int,
                        default=None, help="encoder_depth.")
    parser.add_argument("--activation", type=str, default=None, help="activation.")
    parser.add_argument("--pretrain", type=str,
                        default=None, help="choose pretrain.")
    parser.add_argument("--loss_type", type=str,
                        default=None, help="loss type.")
    parser.add_argument("--use_scheduler", type=bool,
                        default=False, help="use_scheduler.")
    parser.add_argument("--aux", type=bool, default=False,
                        help="choose to do classification")
    parser.add_argument("--attention", type=str, default=None,
                        help="decoder_attention_type.")
    parser.add_argument("--usenorm", type=bool,
                        default=True, help="encoder use bn")
    parser.add_argument("--startpoint", type=int, default=60,
                        help="start cotraining point.")
    parser.add_argument("--classnum", type=int, default=3,
                        help="clf class number.")
    parser.add_argument("--model_file", type=str, default=None, help="model_file")
    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(
        description="train setup for segmentation")
    parser.add_argument(
        "--model_type",
        type=str,
        help="select model type: unet,dcan,dmtn,psinet,convmcd",
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        default="dist_signed",
        help="select distance transform type - dist_mask,dist_contour,dist_signed",
    )
    parser.add_argument("--data_path", type=str, help="path to img jpg files")
    parser.add_argument("--mask_path", type=str, help="path to img jpg files")
    parser.add_argument("--model_file", type=str, help="model_file")
    parser.add_argument("--save_path", type=str, help="results save path.")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument("--encoder", type=str, default=None, help="encoder.")
    parser.add_argument("--encoder_depth", type=int,
                        default=None, help="encoder.")
    parser.add_argument("--activation", type=str, default=None, help="activation.")
    parser.add_argument("--pretrain", type=str,
                        default=None, help="choose pretrain.")
    parser.add_argument("--attention", type=str, default=None,
                        help="decoder_attention_type.")
    parser.add_argument(
        "--val_batch_size", type=int, default=4, help="validation batch size"
    )
    parser.add_argument("--usenorm", type=bool,
                        default=True, help="encoder use bn")
    parser.add_argument("--classnum", type=int, default=3,
                        help="clf class number.")
    return parser
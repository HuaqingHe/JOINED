import cv2
import torch
import os

import torchvision.utils
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
from utils import create_validation_arg_parser, build_model
from skimage import transform, data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd
import surface_distance
import scipy.spatial
import matplotlib
from numpy import mean, std
from utils import *
from losses import *
from sklearn.preprocessing import label_binarize
from dataset import TestDataSet, PreDataSet, CropDataSet
from augmentation import *


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def getJaccard(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    return 1.0 - scipy.spatial.distance.jaccard(testArray, resultArray)


def getPrecisionAndRecall(testImage, resultImage):
    testArray = testImage.flatten()
    resultArray = resultImage.flatten()

    TP = np.sum(testArray * resultArray)
    FP = np.sum((1 - testArray) * resultArray)
    FN = np.sum(testArray * (1 - resultArray))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return precision, recall


def getHD_ASSD(seg_preds, seg_labels):
    label_seg = np.array(seg_labels, dtype=bool)
    predict = np.array(seg_preds, dtype=bool)

    surface_distances = surface_distance.compute_surface_distances(
        label_seg, predict, spacing_mm=(1, 1))

    HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]

    ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) + np.sum(distances_gt_to_pred *
                                                                      surfel_areas_gt)) / (
                   np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))

    return HD, ASSD


def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    if area:
        max_idx = np.argmax(area)
    else:
        return mask_sel
    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel


def evaluate(model, valLoader, device, save_path):
    model.eval()

    name = []
    faz_dice1 = []
    faz_dice2 = []
    faz_jaccard1 = []
    faz_jaccard2 = []
    rv_dice1 = []
    rv_dice2 = []
    rv_jaccard1 = []
    rv_jaccard2 = []
    dice = []
    jaccard = []
    faz_HD = []
    faz_ASSD = []
    rv_HD = []
    rv_ASSD = []
    dicelosses1 = []
    jaccardlosses1 = []
    dicelosses2 = []
    jaccardlosses2 = []

    for i, (inputs, targets) in enumerate(tqdm(valLoader)):
        inputs = inputs.to(device)
        seg_labels = targets.numpy().squeeze()
        targets = targets.to(device)

        outputs = model(inputs)

        seg_outputs = outputs.detach().cpu().numpy().squeeze()
        c, h, w = seg_outputs.shape
        seg_prs = label_binarize(seg_outputs.argmax(
            0).flatten(), classes=[0, 1, 2])
        seg_prs = seg_prs.reshape(h, w, c)
        seg_prs = seg_prs.transpose(2, 0, 1)
        faz_labels = seg_labels[1, :, :]
        faz_prs = seg_prs[1, :, :]
        rv_labels = seg_labels[2, :, :]
        rv_prs = seg_prs[2, :, :]

        faz_dice_1 = f1_score(faz_labels.flatten(),
                              faz_prs.flatten(), average=None)
        faz_dice_2 = getDSC(faz_labels, faz_prs)
        rv_dice_1 = f1_score(rv_labels.flatten(),
                             rv_prs.flatten(), average=None)
        rv_dice_2 = getDSC(rv_labels, rv_prs)
        faz_jaccard_1 = jaccard_score(
            faz_labels.flatten(), faz_prs.flatten(), average=None)
        faz_jaccard_2 = getJaccard(faz_labels, faz_prs)
        rv_jaccard_1 = jaccard_score(
            rv_labels.flatten(), rv_prs.flatten(), average=None)
        rv_jaccard_2 = getJaccard(rv_labels, rv_prs)

        dices, _dicelosses = eval_DiceLoss(mode='multiclass')(torch.from_numpy(
            np.expand_dims(seg_prs, axis=0)), torch.from_numpy(np.expand_dims(seg_labels, axis=0)))
        jaccards, _jaccardlosses = eval_JaccardLoss(mode='multiclass')(torch.from_numpy(np.expand_dims(
            seg_prs, axis=0)), torch.from_numpy(np.expand_dims(seg_labels, axis=0)))

        dices = 1 - dices
        jaccards = 1 - jaccards

        faz_HDs, faz_ASSDs = getHD_ASSD(faz_prs, faz_labels)
        rv_HDs, rv_ASSDs = getHD_ASSD(rv_prs, rv_labels)

        name.append(str(i))
        faz_dice1.append(faz_dice_1)
        faz_dice2.append(faz_dice_2)
        faz_jaccard1.append(faz_jaccard_1)
        faz_jaccard2.append(faz_jaccard_2)
        rv_dice1.append(rv_dice_1)
        rv_dice2.append(rv_dice_2)
        rv_jaccard1.append(rv_jaccard_1)
        rv_jaccard2.append(rv_jaccard_2)
        dice.append(dices.to(torch.float64))
        jaccard.append(jaccards.to(torch.float64))
        faz_HD.append(faz_HDs)
        faz_ASSD.append(faz_ASSDs)
        rv_HD.append(rv_HDs)
        rv_ASSD.append(rv_ASSDs)
        dicelosses1.append(1 - _dicelosses[1])
        jaccardlosses1.append(1 - _jaccardlosses[1])
        dicelosses2.append(1 - _dicelosses[2])
        jaccardlosses2.append(1 - _jaccardlosses[2])

    return name, faz_dice1, faz_dice2, faz_jaccard1, faz_jaccard2, rv_dice1, rv_dice2, rv_jaccard1, rv_jaccard2, dice, jaccard, faz_HD, faz_ASSD, rv_HD, rv_ASSD, dicelosses1, jaccardlosses1, dicelosses2, jaccardlosses2


def predict_mask(model, preLoader, device, save_path, test_list):
    model.eval()

    for i, (inputs, H, W) in enumerate(tqdm(preLoader)):

        inputs = inputs.to(device)
        outputs = model(inputs)

        # torchvision.utils.save_image(outputs, save_path + test_list[i] + 'seg0.bmp')          # torch 保存为图片  看刚分割出来的结果

        seg_outputs = outputs.detach().cpu().numpy().squeeze()  # 这里就不是tensor 已经是在CPU上的降维的ndarray
        # seg_outputs = cv2.resize(seg_outputs, (3, int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        # seg_outputs.resize(3, int(H), int(W))                     # 这里出了一些问题，，debug的时候会卡出来
        # seg_outputs_save = seg_outputs.transpose(1, 2, 0)                                  # 这里看到，存下来并没有啥不同，就是内存变多
        # matplotlib.image.imsave(save_path + test_list[i] + 'seg1.bmp', seg_outputs_save)

        # cv2.imwrite(save_path + test_list[i] + '_0.bmp', seg_outputs[0, :, :] * 255.)
        # cv2.imwrite(save_path + test_list[i] + '_1.bmp', seg_outputs[1, :, :] * 255.)
        # cv2.imwrite(save_path + test_list[i] + '_2.bmp', seg_outputs[2, :, :] * 255.)

        # c, h, w = seg_outputs.shape
        OC_tmp = np.array(seg_outputs[0, :, :] > 0.5, dtype='uint8')
        OC_tmp = find_max_region(OC_tmp)
        # cv2.imwrite(save_path + test_list[i] + '_0.bmp', OC_tmp * 255.)
        OD_tmp = np.array(seg_outputs[1, :, :] > 0.2, dtype='uint8')  # 粗分割搞小一点
        OD_tmp = find_max_region(OD_tmp)
        # cv2.imwrite(save_path + test_list[i] + '_1.bmp', OD_tmp * 255.)
        # BG_tmp = seg_outputs[2, :, :]

        mask_prs = np.full([512, 512], 255, dtype=np.uint8)
        mask_prs[OD_tmp == 1] = 128
        mask_prs[OC_tmp == 1] = 0

        # 老方法，发现在分类的时候会出现一些问题
        # OC_prs = label_binarize(OC_tmp.flatten())        # 这个使用方法在这里不适用，需要修改方法
        # OC_prs = OC_prs.reshape(h, w)
        # cv2.imwrite(save_path + test_list[i] + '_prs0.bmp', OC_prs * 255.)
        #  seg_prs = label_binarize(seg_outputs.argmax(    # seg_outputs.argmax(0)这里会选择值最大的那一层 把
        #      0).flatten(), classes=[0, 1, 2])        # [3， 512， 512] 按照第0维，来进行选择512*512最大的在哪个通道，再平坦成一维，进行三分类的类似onehot编码。
        # seg_prs = seg_prs.reshape(h, w, c)
        # seg_prs = np.array(seg_prs, dtype='uint8')      # 为了能够保存成一个图像
        # cv2.imwriter(save_path + test_list[i] + 'seg.bmp', seg_prs*255)    # 这里发现全是黑的,在乘了255之后还是有值,而且把背景的黑和眼底背景都弄在一起了
        # seg_prs = seg_prs.transpose(2, 0, 1)
        # OC_prs = seg_prs[0, :, :]
        # cv2.imwriter(save_path + test_list[i] + 'OC.bmp', OC_prs)          # 看OC
        # OD_prs = seg_prs[1, :, :]
        # cv2.imwriter(save_path + test_list[i] + 'OD.bmp', OD_prs)          # 看OD
        # BG_prs = seg_prs[2, :, :]
        # cv2.imwriter(save_path + test_list[i] + 'BG.bmp', BG_prs)          # 看BG
        # OC_flag = np.uint8(OC_prs == 1)                 #.astype(int) 这个没有uint8
        # OD_flag = np.uint8(OD_prs == 1)
        # BG_flag = np.uint8(BG_prs == 1)
        # mask_prs = OC_flag*0 + OD_flag*128 + BG_flag*255

        mask_prs = cv2.resize(mask_prs, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        # cv2.imwriter(save_path + test_list[i] + 'seg.bmp', mask_prs)
        # mask_prs = np.uint8(transform.resize(mask_prs, (int(H), int(W))))        # 这里需要变成uint8  但是结果还是全黑的
        im = Image.fromarray(mask_prs)
        im.save(save_path + test_list[i] + '.bmp')

        # 存一个粗分割的512*512的图和mask先训练着
        [X, Y] = np.where(mask_prs == 128)  # 找OD，因为OC不一定找得到
        if X.size == 0:
            x = H / 2
            y = W / 2
        else:
            x = (max(X) + min(X) + 2) / 2  # 找OD中心
            y = (max(Y) + min(Y) + 2) / 2
        a = int(x - 256)  # 找左上角
        b = int(x + 256)
        c = int(y - 256)
        d = int(y + 256)
        # inputs = inputs.detach().cpu().numpy().squeeze()        # 这里的input不是原始尺寸
        # crop_inputs = inputs[0, a:b, c:d]                      # 这里因为输入是CHW 所以要在第一维度加上
        # crop_masks = mask_prs[a:b, c:d]
        # isExists = os.path.exists(save_path + 'crop_img')
        # # 判断结果
        # if not isExists:
        #     os.makedirs(save_path + 'crop_img')
        # isExists = os.path.exists(save_path + 'crop_mask')
        # # 判断结果
        # if not isExists:
        #     os.makedirs(save_path + 'crop_mask')
        # im = Image.fromarray(crop_inputs)
        # im.save(save_path + 'crop_img' + test_list[i] + '.bmp')
        # cv2.imwriter(save_path + 'crop_mask' + test_list[i] + '.bmp', crop_masks)

    return 0


def predict_mask_vote(model1, model2, model3, model4, model5, preLoader, device, save_path, test_list):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    for i, (inputs, H, W) in enumerate(tqdm(preLoader)):

        inputs = inputs.to(device)
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)
        outputs3 = model3(inputs)
        outputs4 = model4(inputs)
        outputs5 = model5(inputs)
        seg_outputs1 = outputs1.detach().cpu().numpy().squeeze()  # 这里就不是tensor 已经是在CPU上的降维的ndarray
        seg_outputs2 = outputs2.detach().cpu().numpy().squeeze()
        seg_outputs3 = outputs3.detach().cpu().numpy().squeeze()
        seg_outputs4 = outputs4.detach().cpu().numpy().squeeze()
        seg_outputs5 = outputs5.detach().cpu().numpy().squeeze()
        OC_tmp = np.array((seg_outputs1[0, :, :] + seg_outputs2[0, :, :] + seg_outputs3[0, :, :] + seg_outputs4[0, :, :]
                           + seg_outputs5[0, :, :]) / 5 > 0.5, dtype='uint8')
        OD_tmp = np.array((seg_outputs1[1, :, :] + seg_outputs2[1, :, :] + seg_outputs3[1, :, :] + seg_outputs4[1, :, :]
                           + seg_outputs5[1, :, :]) / 5 > 0.5, dtype='uint8')
        mask_prs = np.full([512, 512], 255, dtype=np.uint8)
        mask_prs[OD_tmp == 1] = 128
        mask_prs[OC_tmp == 1] = 0
        mask_prs = cv2.resize(mask_prs, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        im = Image.fromarray(mask_prs)  # 其实有把RGB变成BGR的问题，显示就是红变成蓝色，但是这里是个灰度图就没问题
        im.save(save_path + test_list[i] + '.bmp')

        # 存一个粗分割的512*512的图和mask先训练着
        [X, Y] = np.where(mask_prs == 128)  # 找OD，因为OC不一定找得到
        if X.size == 0:
            x = H / 2
            y = W / 2
        else:
            x = (max(X) + min(X) + 2) / 2  # 找OD中心
            y = (max(Y) + min(Y) + 2) / 2
        a = int(x - 256)  # 找左上角
        b = int(x + 256)
        c = int(y - 256)
        d = int(y + 256)
        # inputs = inputs.detach().cpu().numpy().squeeze()      #因为input不是原始尺寸
        # crop_inputs = inputs[a:b, c:d]
        # crop_masks = mask_prs[a:b, c:d]
        # isExists = os.path.exists(save_path + 'crop_img')
        # # 判断结果
        # if not isExists:
        #     os.makedirs(save_path + 'crop_img')
        # isExists = os.path.exists(save_path + 'crop_mask')
        # # 判断结果
        # if not isExists:
        #     os.makedirs(save_path + 'crop_mask')
        # im = Image.fromarray(crop_inputs)
        # im.save(save_path + 'crop_img' + test_list[i] + '.bmp')
        # cv2.imwriter(save_path + 'crop_mask' + test_list[i] + '.bmp', crop_masks)
    return 0


def crop_img(model, preLoader, device, save_path, test_list):
    size = 448
    for i, (inputs, masks, H, W) in enumerate(tqdm(preLoader)):
        # inputs = inputs.to(device)
        # outputs = model(inputs)
        inputs = inputs.detach().cpu().numpy().squeeze()
        masks = masks.detach().numpy().squeeze()  # 变成ndarray好找中心
        [X, Y] = np.where(masks == 128)  # 找OD，因为OC不一定找得到
        if X.size == 0:
            x = H / 2
            y = W / 2
        else:
            x = (max(X) + min(X) + 2) / 2  # 找OD中心
            y = (max(Y) + min(Y) + 2) / 2
        a = int(x - size/2)  # 找左上角
        b = int(x + size/2)
        c = int(y - size/2)
        d = int(y + size/2)
        # crop_trans = A.Compose([
        #     # A.augmentations.crops.transforms.Crop(x_min=a, y_min=c, x_max=b, y_max=d),             # crop 512
        #     ToTensorV2()
        # ])
        crop_inputs = inputs[a:b, c:d]

        # isExists = os.path.exists(save_path + 'crop' + str(size))
        # # 判断结果
        # if not isExists:
        #     os.makedirs(save_path + 'crop' + str(size))
        #
        # cv2.imwrite(save_path + 'crop' + str(size) + '/' + test_list[i] + 'crop.bmp', crop_inputs)
        # 把crop的图片和mask存下来用于精分割训练，在test的时候不需要
        crop_masks = masks[a:b, c:d]
        isExists = os.path.exists(save_path + 'crop_img' + str(size))
        # 判断结果
        if not isExists:
            os.makedirs(save_path + 'crop_img' + str(size))
        isExists = os.path.exists(save_path + 'crop_mask' + str(size))
        # 判断结果
        if not isExists:
            os.makedirs(save_path + 'crop_mask' + str(size))
        cv2.imwrite(save_path + 'crop_img' + str(size) + '/' + test_list[i] + '.bmp', crop_inputs)
        cv2.imwrite(save_path + 'crop_mask' + str(size) + '/' + test_list[i] + '.bmp', crop_masks)

        # totens_trans = transforms.ToTensor()
        # crop_inputs = totens_trans(crop_inputs)
        # # crop_inputs = torch.tensor(crop_inputs)
        # crop_inputs = torch.reshape(crop_inputs, (1, 3, size, size))
        # crop_inputs = crop_inputs.to(device)
        # outputs = model(crop_inputs)
        # # torchvision.utils.save_image(crop_inputs, save_path + 'crop' + str(size) + '/' + test_list[i] + 'crop0.bmp')
        # torchvision.utils.save_image(outputs, save_path + 'crop' + str(size) + '/' + test_list[i] + 'cropseg0.bmp')
        #
        # # 这里再把crop之后的分割放回原来的位置去
        # seg_outputs = outputs.detach().cpu().numpy().squeeze()
        # OC_tmp = np.array(seg_outputs[0, :, :] > 0.5, dtype='uint8')
        # OC_tmp = find_max_region(OC_tmp)
        # cv2.imwrite(save_path + 'crop' + str(size) + '/' + test_list[i] + '_0.bmp', OC_tmp * 255.)
        # OD_tmp = np.array(seg_outputs[1, :, :] > 0.5, dtype='uint8')
        # OD_tmp = find_max_region(OD_tmp)
        # cv2.imwrite(save_path + 'crop' + str(size) + '/' + test_list[i] + '_1.bmp', OD_tmp * 255.)
        # # BG_tmp = seg_outputs[2, :, :]
        #
        # mask_prs = np.full([int(H), int(W)], 255, dtype=np.uint8)
        # mask_crop = np.full([size, size], 255, dtype=np.uint8)
        # mask_crop[OD_tmp == 1] = 128
        # mask_crop[OC_tmp == 1] = 0
        # mask_prs[a:b, c:d] = mask_crop
        # im = Image.fromarray(mask_prs)
        # im.save(save_path + 'crop' + str(size) + '/' + test_list[i] + '.bmp')
    return 0



def crop_img_vote(model1, model2, model3, model4, model5, preLoader, device, save_path, test_list):
    for i, (inputs, masks, H, W) in enumerate(tqdm(preLoader)):
        # inputs = inputs.to(device)
        # outputs = model(inputs)
        inputs = inputs.detach().cpu().numpy().squeeze()
        masks = masks.detach().numpy().squeeze()  # 变成ndarray好找中心
        [X, Y] = np.where(masks == 128)  # 找OD，因为OC不一定找得到
        if X.size == 0:
            x = H / 2
            y = W / 2
        else:
            x = (max(X) + min(X) + 2) / 2  # 找OD中心
            y = (max(Y) + min(Y) + 2) / 2
        a = int(x - 256)  # 找左上角
        b = int(x + 256)
        c = int(y - 256)
        d = int(y + 256)

        crop_inputs = inputs[a:b, c:d]

        totens_trans = transforms.ToTensor()
        crop_inputs = totens_trans(crop_inputs)
        crop_inputs = torch.reshape(crop_inputs, (1, 3, 512, 512))
        crop_inputs = crop_inputs.to(device)

        outputs1 = model1(crop_inputs)
        outputs2 = model2(crop_inputs)
        outputs3 = model3(crop_inputs)
        outputs4 = model4(crop_inputs)
        outputs5 = model5(crop_inputs)
        seg_outputs1 = outputs1.detach().cpu().numpy().squeeze()  # 这里就不是tensor 已经是在CPU上的降维的ndarray
        seg_outputs2 = outputs2.detach().cpu().numpy().squeeze()
        seg_outputs3 = outputs3.detach().cpu().numpy().squeeze()
        seg_outputs4 = outputs4.detach().cpu().numpy().squeeze()
        seg_outputs5 = outputs5.detach().cpu().numpy().squeeze()
        OC_tmp = np.array((seg_outputs1[0, :, :] + seg_outputs2[0, :, :] + seg_outputs3[0, :, :] + seg_outputs4[0, :, :]
                           + seg_outputs5[0, :, :]) / 5 > 0.5, dtype='uint8')
        OD_tmp = np.array((seg_outputs1[1, :, :] + seg_outputs2[1, :, :] + seg_outputs3[1, :, :] + seg_outputs4[1, :, :]
                           + seg_outputs5[1, :, :]) / 5 > 0.5, dtype='uint8')
        isExists = os.path.exists(save_path + 'crop2')
        # 判断结果
        if not isExists:
            os.makedirs(save_path + 'crop2')
        # torchvision.utils.save_image(outputs, save_path + 'crop2/' + test_list[i] + 'cropseg0.bmp')

        # 这里再把crop之后的分割放回原来的位置去

        OC_tmp = find_max_region(OC_tmp)
        cv2.imwrite(save_path + 'crop2/' + test_list[i] + '_0.bmp', OC_tmp * 255.)
        OD_tmp = find_max_region(OD_tmp)
        cv2.imwrite(save_path + 'crop2/' + test_list[i] + '_1.bmp', OD_tmp * 255.)
        # BG_tmp = seg_outputs[2, :, :]

        mask_prs = np.full([int(H), int(W)], 255, dtype=np.uint8)
        mask_crop = np.full([512, 512], 255, dtype=np.uint8)
        mask_crop[OD_tmp == 1] = 128
        mask_crop[OC_tmp == 1] = 0
        mask_prs[a:b, c:d] = mask_crop
        im = Image.fromarray(mask_prs)
        im.save(save_path + 'crop2/' + test_list[i] + '.bmp')
    return 0


def main():
    args = create_validation_arg_parser().parse_args()
    args.usenorm = True
    args.classnum = 3
    args.loss = 'dice'
    args.pretrain = 'imagenet'
    # args.model_type = 'unet_smp'
    args.model_type = 'unet++'
    # args.encoder = 'timm-resnest50d'
    args.encoder = 'timm-efficientnet-b4'
    args.encoder_depth = 4
    args.activation = 'sigmoid'
    args.data_path = r'E:\MICCAI2021\code\data/'
    # args.model_file = r'model/dics_cup_efficien_unet/1/best_dice_0.9075OCDice_0.8685ODDice_0.9464.pt'          # model1
    args.model_file = r'model/disc_cup_efficien_unet++/model1/best_dice_0.9062OCDice_0.8777ODDice_0.9348.pt'       # model1-2
    # args.model_file = r'model/model1/200dice_0.9161OCDice_0.8772ODDice_0.9549.pt'            # model1-3
    # args.model_file = r'model/dics_cup_efficien_unet/2/100dice_0.9122OCDice_0.8826ODDice_0.9418.pt'          # model2
    # args.model_file = r'model/disc_cup_efficien_unet++/model2/best_dice_0.9089OCDice_0.8733ODDice_0.9446.pt'  # model2-2
    # args.model_file = r'model/dics_cup_efficien_unet/3/100dice_0.8882OCDice_0.8335ODDice_0.943.pt'            # model3
    # args.model_file = r'model/disc_cup_efficien_unet++/model3/100dice_0.8891OCDice_0.8317ODDice_0.9466.pt'               # model3-2
    # args.model_file = r'model/dics_cup_efficien_unet/4/best_dice_0.9043OCDice_0.8699ODDice_0.9387.pt'            # model4
    # args.model_file = r'model/disc_cup_efficien_unet++/model4/best_dice_0.903OCDice_0.8662ODDice_0.9399.pt'            # model4-2
    # args.model_file = r'model/dics_cup_efficien_unet/5/best_dice_0.9159OCDice_0.8818ODDice_0.95.pt'           # model5
    # args.model_file = r'model/disc_cup_efficien_unet++/model5/best_dice_0.9172OCDice_0.8857ODDice_0.9488.pt'            # model5-2 分数不那么高，感觉过拟合了
    # args.model_file = r'E:\MICCAI2021\code\50dice_0.878jaccard_0.783.pt'      # OD 是个环
    args.save_path = r'result2/efficient++/mask1/'
    args.mask_path = args.save_path  # 用跑出来的mask来crop区域
    args.batch_size = 64

    data_path = args.data_path
    mask_path = args.mask_path
    model_file = args.model_file
    # model_file1 = r'model/dics_cup_efficien_unet/1/best_dice_0.9075OCDice_0.8685ODDice_0.9464.pt'  # model1
    # model_file2 = r'model/dics_cup_efficien_unet/2/100dice_0.9122OCDice_0.8826ODDice_0.9418.pt'  # model2
    # model_file3 = r'model/dics_cup_efficien_unet/3/100dice_0.8882OCDice_0.8335ODDice_0.943.pt'  # model3
    # model_file4 = r'model/dics_cup_efficien_unet/4/best_dice_0.9043OCDice_0.8699ODDice_0.9387.pt'  # model4
    # model_file5 = r'model/dics_cup_efficien_unet/5/best_dice_0.9159OCDice_0.8818ODDice_0.95.pt'  # model5
    model_fine_file = r'model/disc_cup_efficien_unet++/fine/416/1/300dice_0.9251OCDice_0.8896ODDice_0.9605.pt'
    model_fine_file1 = r'model/dics_cup_efficien_unet/fine/1/best_dice_0.9226OCDice_0.8942ODDice_0.951.pt'
    model_fine_file2 = r'model/dics_cup_efficien_unet/fine/2/best_dice_0.9125OCDice_0.8719ODDice_0.9531.pt'
    model_fine_file3 = r'model/dics_cup_efficien_unet/fine/3/300dice_0.8909OCDice_0.834ODDice_0.9479.pt'
    model_fine_file4 = r'model/dics_cup_efficien_unet/fine/4/best_dice_0.9044OCDice_0.8589ODDice_0.9499.pt'
    model_fine_file5 = r'model/dics_cup_efficien_unet/fine/5/best_dice_0.9203OCDice_0.8863ODDice_0.9542.pt'

    save_path = args.save_path

    # distance_type = args.distance_type
    # print(distance_type)

    names = []
    faz_dice1 = []
    faz_dice2 = []
    faz_jaccard1 = []
    faz_jaccard2 = []
    rv_dice1 = []
    rv_dice2 = []
    rv_jaccard1 = []
    rv_jaccard2 = []
    dice = []
    jaccard = []
    faz_HD = []
    faz_ASSD = []
    rv_HD = []
    rv_ASSD = []
    dicelosses1 = []
    jaccardlosses1 = []
    dicelosses2 = []
    jaccardlosses2 = []

    pretrain = args.pretrain
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.batch_size = 1
    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    # 6M
    train_tmp_list = list(range(1, 101))
    train_num_list = [str(x).zfill(4) for x in train_tmp_list]
    test_tmp_list = list(range(101, 201))
    test_num_list = [str(x).zfill(4) for x in test_tmp_list]

    # 3M
    # train_num_list = list(range(10301, 10441))
    # test_num_list = list(range(10451, 10501))

    m, s = mean_and_std(args.data_path, train_num_list)

    test_data = TestDataSet(data_path=args.data_path, mask_path=args.mask_path,
                            transform=val_crop_aug(m, s), num_list=test_num_list, class_num=args.classnum)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model_type, args.encoder,
                        pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    model_fine = build_model(args.model_type, args.encoder,
                             pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)

    model.load_state_dict(torch.load(model_file, 'cuda:0'))
    model_fine.load_state_dict(torch.load(model_fine_file, 'cuda:0'))

    # coarse vote
    # model1 = build_model(args.model_type, args.encoder,
    #                      pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model1.load_state_dict(torch.load(model_file1, 'cuda:0'))
    #
    # model2 = build_model(args.model_type, args.encoder,
    #                      pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model2.load_state_dict(torch.load(model_file2, 'cuda:0'))
    #
    # model3 = build_model(args.model_type, args.encoder,
    #                      pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model3.load_state_dict(torch.load(model_file3, 'cuda:0'))
    # model4 = build_model(args.model_type, args.encoder,
    #                      pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model4.load_state_dict(torch.load(model_file4, 'cuda:0'))
    # model5 = build_model(args.model_type, args.encoder,
    #                      pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model5.load_state_dict(torch.load(model_file5, 'cuda:0'))

    # fine_vote
    # model_fine1 = build_model(args.model_type, args.encoder,
    #                           pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model_fine1.load_state_dict(torch.load(model_fine_file1, 'cuda:0'))
    #
    # model_fine2 = build_model(args.model_type, args.encoder,
    #                           pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model_fine2.load_state_dict(torch.load(model_fine_file2, 'cuda:0'))
    #
    # model_fine3 = build_model(args.model_type, args.encoder,
    #                           pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model_fine3.load_state_dict(torch.load(model_fine_file3, 'cuda:0'))
    # model_fine4 = build_model(args.model_type, args.encoder,
    #                           pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model_fine4.load_state_dict(torch.load(model_fine_file4, 'cuda:0'))
    # model_fine5 = build_model(args.model_type, args.encoder,
    #                           pretrain, args.classnum, args.encoder_depth, args.activation, aux=False).to(device)
    # model_fine5.load_state_dict(torch.load(model_fine_file5, 'cuda:0'))
    ### change
    pre_data = PreDataSet(data_path=args.data_path, mask_path=args.mask_path,
                          transform=val_crop_aug(m, s), num_list=test_num_list, class_num=args.classnum)

    crop_data = CropDataSet(data_path=args.data_path, mask_path=args.mask_path,
                            transform=None, num_list=test_num_list, class_num=args.classnum)
    # 用于trainimg的crop 需要把mask也crop出来
    args.mask_path = r'data/mask/'
    save_path =  r'data/'
    crop_traindata = CropDataSet(data_path=args.data_path, mask_path=args.mask_path,
                            transform=None, num_list=train_num_list, class_num=args.classnum)
    crop_trainloader = torch.utils.data.DataLoader(
        crop_traindata, batch_size=1, shuffle=False)

    pre_loader = torch.utils.data.DataLoader(
        pre_data, batch_size=1, shuffle=False)
    crop_loader = torch.utils.data.DataLoader(
        crop_data, batch_size=1, shuffle=False)
    # flag = predict_mask(model, pre_loader, device, save_path, test_num_list)      # 预测mask
    flag = crop_img(model, crop_trainloader, device, save_path, train_num_list)      # 将1-100的od部分分割出来进行训练
    flag = crop_img(model_fine, crop_loader, device, save_path, test_num_list)  # 粗crop
    # flag = predict_mask_vote(model1, model2, model3, model4, model5, pre_loader, device, save_path, test_num_list)
    # flag = crop_img_vote(model_fine1, model_fine2, model_fine3, model_fine4, model_fine5, crop_loader, device,
    #                      save_path, test_num_list)  # 粗crop

    name, _faz_dice1, _faz_dice2, _faz_jaccard1, _faz_jaccard2, _rv_dice1, _rv_dice2, _rv_jaccard1, _rv_jaccard2, _dice, _jaccard, _faz_HD, _faz_ASSD, _rv_HD, _rv_ASSD, _dicelosses1, _jaccardlosses1, _dicelosses2, _jaccardlosses2 = evaluate(
        model, test_loader, device, save_path)

    names.extend(name)
    faz_dice1.extend(_faz_dice1)
    faz_dice2.extend(_faz_dice2)
    faz_jaccard1.extend(_faz_jaccard1)
    faz_jaccard2.extend(_faz_jaccard2)
    rv_dice1.extend(_rv_dice1)
    rv_dice2.extend(_rv_dice2)
    rv_jaccard1.extend(_rv_jaccard1)
    rv_jaccard2.extend(_rv_jaccard2)
    dice.extend(_dice)
    jaccard.extend(_jaccard)
    faz_HD.extend(_faz_HD)
    faz_ASSD.extend(_faz_ASSD)
    rv_HD.extend(_rv_HD)
    rv_ASSD.extend(_rv_ASSD)
    dicelosses1.extend(_dicelosses1)
    jaccardlosses1.extend(_jaccardlosses1)
    dicelosses2.extend(_dicelosses2)
    jaccardlosses2.extend(_jaccardlosses2)

    dataframe = pd.DataFrame({'case': names,
                              'faz_dice1': faz_dice1, 'faz_dice2': faz_dice2,
                              'faz_jaccard1': faz_jaccard1, 'faz_jaccard2': faz_jaccard2,
                              'rv_dice1': rv_dice1, 'rv_dice2': rv_dice2,
                              'rv_jaccard1': rv_jaccard1, 'rv_jaccard2': rv_jaccard2,
                              'dice': dice, 'jaccard': jaccard,
                              'faz_HD': faz_HD, 'faz_ASSD': faz_ASSD,
                              'rv_HD': rv_HD, 'rv_ASSD': rv_ASSD,
                              'dicelosses1': dicelosses1, 'jaccardlosses1': jaccardlosses1,
                              'dicelosses2': dicelosses2, 'jaccardlosses2': jaccardlosses2,
                              })
    dataframe.to_csv(save_path + "/detail_metrics.csv",
                     index=False, sep=',')
    print('Counting CSV generated!')
    mean_resultframe = pd.DataFrame({
        'faz_dice': mean(faz_dice2), 'faz_jaccard': mean(faz_jaccard2),
        'rv_dice': mean(rv_dice2), 'rv_jaccard': mean(rv_jaccard2),
        'dice': mean(dice), 'jaccard': mean(jaccard),
        'faz_HD': mean(faz_HD), 'faz_ASSD': mean(faz_ASSD),
        'rv_HD': mean(rv_HD), 'rv_ASSD': mean(rv_ASSD)}, index=[1])
    mean_resultframe.to_csv(save_path + "/mean_metrics.csv", index=0)
    std_resultframe = pd.DataFrame({
        'faz_dice': std(faz_dice2, ddof=1), 'faz_jaccard': std(faz_jaccard2, ddof=1),
        'rv_dice': std(rv_dice2, ddof=1), 'rv_jaccard': std(rv_jaccard2, ddof=1),
        'dice': std(dice, ddof=1), 'jaccard': std(jaccard, ddof=1),
        'faz_HD': std(faz_HD, ddof=1), 'faz_ASSD': std(faz_ASSD, ddof=1),
        'rv_HD': std(rv_HD, ddof=1), 'rv_ASSD': std(rv_ASSD, ddof=1)}, index=[1])
    std_resultframe.to_csv(save_path + "/std_metrics.csv", index=0)
    print('Calculating CSV generated!')


if __name__ == "__main__":
    main()

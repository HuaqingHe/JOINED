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


def getvCDR(preds_DC):
    dim = preds_DC.size(0)
    vCDR = []
    mask_tmp = np.round(preds_DC.detach().cpu().numpy().squeeze())
    # for i in range(dim):
    OC_tmp = mask_tmp[0, :, :]
    OC_tmp = np.round(OC_tmp)
    [X, Y] = np.where(OC_tmp == 1)
    if Y.size == 0:
        Y_OC = 0
    else:
        Y_OC = max(Y) - min(Y)
    
    OD_tmp = mask_tmp[1, :, :]
    OD_tmp = np.round(OD_tmp)
    [X, Y] = np.where(OD_tmp == 1)
    if Y.size == 0:
        tmp = 0
        vCDR.append(tmp)
    else:
        Y_OD = max(Y) - min(Y)
        vCDR.append(Y_OC/Y_OD)
    vCDR = torch.from_numpy(np.expand_dims(np.array(vCDR), 1))
    return vCDR

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


def evaluate(model, valLoader, device, save_path, test_list):
    model.eval()
    torch.set_grad_enabled(False)
    name = []
    OD_dice = []
    OC_dice = []
    OD_jaccard = []
    OC_jaccard = []
    dice = []
    jaccard = []
    vCDR = []
    vCDR_tar = []
    vCDRlosses = []
    dicelossesOD = []
    jaccardlossesOD = []
    dicelossesOC = []
    jaccardlossesOC = []
    criterion = [
            weighted_define_loss('dice', 3),
            train_DiceLoss_weighted(mode='multiclass'),
            train_JaccardLoss_weighted(mode='multiclass')
    ]
    dist_criterion = smp.utils.losses.MSELoss()
    loss_criterion, dice_criterion, jaccard_criterion = criterion[
            0], criterion[1], criterion[2]
    for i, (inputs, targets) in enumerate(tqdm(valLoader)):
        inputs = inputs.to(device)
        seg_labels = targets.numpy().squeeze()
        targets = targets.to(device)
        c, h, w = seg_labels.shape
        
        outputs = model(inputs)
        seg_outputs = outputs.detach().cpu().numpy().squeeze()
        OC_tmp = np.array(seg_outputs[0, :, :] > 0.5, dtype='uint8')
        OC_tmp = find_max_region(OC_tmp)
        OC_tmp = 1 - find_max_region(1 - OC_tmp)

        OD_tmp = np.array(seg_outputs[1, :, :] > 0.5, dtype='uint8')  # 粗分割搞小一点
        OD_tmp = find_max_region(OD_tmp)
        OD_tmp = 1 - find_max_region(1 - OD_tmp)

        BG_tmp = np.full([h, w], 1, dtype=np.uint8)
        BG_tmp[OD_tmp == 1] = 0
        BG_tmp[OC_tmp == 1] = 0

        mask_prs = np.full([h, w], 255, dtype=np.uint8)
        mask_prs[OD_tmp == 1] = 128
        mask_prs[OC_tmp == 1] = 0
        im = Image.fromarray(mask_prs)
        im.save(save_path + test_list[i] + '.bmp')

        
        pred_DC = np.array([OC_tmp, OD_tmp, BG_tmp])
        pred_DC = np.expand_dims(pred_DC, axis=0)
        pred_DC = torch.from_numpy(pred_DC)
        # outputs dice
        dicelossesODs, loss = loss_criterion(outputs, targets.to(torch.int64))
        dicelossesOCs = loss*2 - dicelossesODs
        # predict dice
        _OD_dice, _dice = dice_criterion(pred_DC.to(device), targets.to(torch.int64))
        _dice = 1 - _dice
        _OD_dice = 1 - _OD_dice
        _OC_dice = _dice*2 - _OD_dice
        # outputs jaccard
        jaccardlossesODs, loss = loss_criterion(outputs, targets.to(torch.int64))
        jaccardlossesOCs = loss*2 - jaccardlossesODs
        # predict jaccard
        _OD_jaccard, _jaccard = jaccard_criterion(pred_DC.to(device), targets.to(torch.int64))
        _jaccard = 1 - _jaccard
        _OD_jaccard = 1 - _OD_jaccard
        _OC_jaccard = _jaccard*2 - _OD_jaccard
        # vCDR and vCDRloss    
        vCDRs = getvCDR(pred_DC)
        vCDRs_tar = getvCDR(targets)
        vCDRlosses_s = dist_criterion(vCDRs, vCDRs_tar)
        vCDRlosses_s = vCDRlosses_s.cpu().numpy().squeeze() ** 0.5
        name.append(test_list[i])
        OD_dice.append(_OD_dice.cpu().numpy())
        OC_dice.append(_OC_dice.cpu().numpy())
        OD_jaccard.append(_OD_jaccard.cpu().numpy())
        OC_jaccard.append(_OC_jaccard.cpu().numpy())
        dice.append(_dice.cpu().numpy())
        jaccard.append(_jaccard.cpu().numpy())
        vCDR.append(vCDRs.cpu().numpy().squeeze())
        vCDR_tar.append(vCDRs_tar.cpu().numpy().squeeze())
        vCDRlosses.append(vCDRlosses_s)
        dicelossesOD.append(dicelossesODs.cpu().numpy())
        jaccardlossesOD.append(jaccardlossesODs.cpu().numpy())
        dicelossesOC.append(dicelossesOCs.cpu().numpy())
        jaccardlossesOC.append(jaccardlossesOCs.cpu().numpy())

    return name, OD_dice, OC_dice, OD_jaccard, OC_jaccard, dice, jaccard, vCDR, vCDR_tar, vCDRlosses, dicelossesOD, jaccardlossesOD, dicelossesOC, jaccardlossesOC
    #      name, _OD_dice, _OC_dice, _OD_jaccard, _OC_jaccard,  _dice, _jaccard, _vCDR, _vCDRlosses, _dicelossesOD, _jaccardlossesOD, _dicelossesOC, _jaccardlossesOC

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

        
    return 0




def main():
    args = create_validation_arg_parser().parse_args()
    args.usenorm = True
    args.classnum = 3
    args.loss = 'dice'
    args.pretrain = 'imagenet'
    args.encoder_depth = 4
    args.activation = 'sigmoid'
    args.data_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save10fovea=0.05hDist=1/result2/crop/fundu448/'
    args.mask_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save10fovea=0.05hDist=1/result2/crop/mask448/'
    args.batch_size = 1
    args.cuda_no = 0
    args.save_path = '/raid5/hehq/MICCAI2021/ISBI/segmentation_fovea_ODOC/model7_1/fineODOC448+timm-efficientnet-b4+unet_smp/result5_1/'
    args.encoder = 'timm-efficientnet-b4'
    # args.encoder = 'timm-resnest50d'
    # args.model_type = 'unet++'
    args.model_type = 'unet_smp'
    model_file_list = [ # Loss 加上了 vCDR
        '/raid5/hehq/MICCAI2021/ISBI/segmentation_fovea_ODOC/model7_1/fineODOC448+timm-efficientnet-b4+unet_smp/1/best_dice_0.9208OCDice_0.8843ODDice_0.9572MAE_vCDR_0.0455.pt',
        '/raid5/hehq/MICCAI2021/ISBI/segmentation_fovea_ODOC/model7_1/fineODOC448+timm-efficientnet-b4+unet_smp/2/best_dice_0.8896OCDice_0.845ODDice_0.9342MAE_vCDR_0.0745.pt',
        '/raid5/hehq/MICCAI2021/ISBI/segmentation_fovea_ODOC/model7_1/fineODOC448+timm-efficientnet-b4+unet_smp/3/best_dice_0.8788OCDice_0.8358ODDice_0.9217MAE_vCDR_0.1079.pt',
        '/raid5/hehq/MICCAI2021/ISBI/segmentation_fovea_ODOC/model7_1/fineODOC448+timm-efficientnet-b4+unet_smp/4/best_dice_0.9074OCDice_0.8636ODDice_0.9512MAE_vCDR_0.0685.pt',
        '/raid5/hehq/MICCAI2021/ISBI/segmentation_fovea_ODOC/model7_1/fineODOC448+timm-efficientnet-b4+unet_smp/5/best_dice_0.9209OCDice_0.885ODDice_0.9569MAE_vCDR_0.0551.pt'
    ]
    save_path = args.save_path


    names_all = []
    OD_dice_all = []
    OC_dice_all = []
    OD_jaccard_all = []
    OC_jaccard_all = []
    vCDR_all = []
    vCDR_tar_all = []
    vCDRlosses_all = []
    dice_all = []
    jaccard_all = []
    dicelossesOD_all = []
    jaccardlossesOD_all = []
    dicelossesOC_all = []
    jaccardlossesOC_all = []

    pretrain = args.pretrain
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.batch_size = 1
    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    # # 6M
    # train_tmp_list = list(range(1, 101))
    # train_num_list = [str(x).zfill(4) for x in train_tmp_list]
    # test_tmp_list = list(range(101, 201))
    # test_num_list = [str(x).zfill(4) for x in test_tmp_list]

    for flod in [4, 3, 2, 1, 0]:
        names = []
        OD_dice = []
        OC_dice = []
        OD_jaccard = []
        OC_jaccard = []
        vCDR = []
        vCDR_tar = []
        vCDRlosses = []
        dice = []
        jaccard = []
        dicelossesOD = []
        jaccardlossesOD = []
        dicelossesOC = []
        jaccardlossesOC = []
        if flod == 0:  # flod1
            # train_file_names = train_file_names[0:80]
            # val_file_names = val_file_names[80:100]
            train_tmp_list = list(range(1, 81))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            test_num_list = [str(x).zfill(4) for x in range(81, 101)]
            
        elif flod == 1:  # flod2
            # train_file_names = train_file_names[0:60] + train_file_names[80:100]
            # val_file_names = val_file_names[60:80]
            train_tmp_list = list(range(1, 61)) + list(range(81, 101))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            test_num_list = [str(x).zfill(4) for x in range(61, 81)]
            
        elif flod == 2:  # flod3
            # train_file_names = train_file_names[0:40] + train_file_names[60:100]
            # val_file_names = val_file_names[40:60]
            train_tmp_list = list(range(1, 41)) + list(range(61, 101))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            test_num_list = [str(x).zfill(4) for x in range(41, 61)]
            
        elif flod == 3:  # flod4
            # train_file_names = train_file_names[0:20] + train_file_names[40:100]
            # val_file_names = val_file_names[20:40]
            train_tmp_list = list(range(1, 21)) + list(range(41, 101))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            test_num_list = [str(x).zfill(4) for x in range(21, 41)]
            
        else:  # flod5
            # train_file_names = train_file_names[20:100]
            # val_file_names = val_file_names[0:20]
            train_tmp_list = list(range(21, 101))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            test_num_list = [str(x).zfill(4) for x in range(1, 21)]
        model_file = model_file_list[flod]
        m, s = mean_and_std(args.data_path, train_num_list)

        test_data = TestDataSet(data_path=args.data_path, mask_path=args.mask_path,
                            transform=val_crop_aug(m, s), num_list=test_num_list, class_num=args.classnum)
        test_loader = torch.utils.data.DataLoader(
                            test_data, batch_size=args.batch_size, shuffle=False)

        model = build_model(args.model_type, args.encoder,
                            pretrain, args.classnum, args.encoder_depth, args.activation, in_channel=5, aux=False).to(device)
        model.load_state_dict(torch.load(model_file))
        model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7
        name, _OD_dice, _OC_dice, _OD_jaccard, _OC_jaccard,  _dice, _jaccard, _vCDR, _vCDR_tar, _vCDRlosses, _dicelossesOD, _jaccardlossesOD, _dicelossesOC, _jaccardlossesOC = evaluate(
                        model, test_loader, device, save_path, test_num_list)
        
        names.extend(name)
        OD_dice.extend(_OD_dice)
        OC_dice.extend(_OC_dice)
        OD_jaccard.extend(_OD_jaccard)
        OC_jaccard.extend(_OC_jaccard)
        vCDR.extend(_vCDR)
        vCDR_tar.extend(_vCDR_tar)
        vCDRlosses.extend(_vCDRlosses)
        dice.extend(_dice)
        jaccard.extend(_jaccard)
        dicelossesOD.extend(_dicelossesOD)
        jaccardlossesOD.extend(_jaccardlossesOD)
        dicelossesOC.extend(_dicelossesOC)
        jaccardlossesOC.extend(_jaccardlossesOC)

        dataframe = pd.DataFrame({'case': names,
                              'OD_dice': OD_dice, 'OC_dice': OC_dice,
                              'OD_jaccard': OD_jaccard, 'OC_jaccard': OC_jaccard,
                              'dice': dice, 'jaccard': jaccard, 
                              'vCDR': vCDR, 'vCDR_real': vCDR_tar, 'vCDRloss': vCDRlosses,
                              'dicelossesOD': dicelossesOD, 'jaccardlossesOD': jaccardlossesOD,
                              'dicelossesOC': dicelossesOC, 'jaccardlossesOC': jaccardlossesOC,
                              })
        dataframe.to_csv(save_path + '_' + str(flod+1) + "detail_metrics.csv",
                     index=False, sep=',')
        print('_' + str(flod+1) + 'Counting CSV generated!')
        mean_resultframe = pd.DataFrame({
            'OD_dice': mean(OD_dice), 'OC_dice': mean(OC_dice),
            'OD_jaccard': mean(OD_jaccard), 'OC_jaccard': mean(OC_jaccard),
            'dice': mean(dice), 'jaccard': mean(jaccard),
            'vCDR': mean(vCDR), 'vCDR_real': mean(vCDR_tar), 'vCDRloss': mean(vCDRlosses)}, index=[1])
        mean_resultframe.to_csv(save_path + '_' + str(flod+1) + "mean_metrics.csv", index=0)
        print('_' + str(flod+1) + 'Calculating CSV generated!')

        names_all.extend(name)
        OD_dice_all.extend(_OD_dice)
        OC_dice_all.extend(_OC_dice)
        OD_jaccard_all.extend(_OD_jaccard)
        OC_jaccard_all.extend(_OC_jaccard)
        vCDR_all.extend(_vCDR)
        vCDR_tar_all.extend(_vCDR_tar)
        vCDRlosses_all.extend(_vCDRlosses)
        dice_all.extend(_dice)
        jaccard_all.extend(_jaccard)
        dicelossesOD_all.extend(_dicelossesOD)
        jaccardlossesOD_all.extend(_jaccardlossesOD)
        dicelossesOC_all.extend(_dicelossesOC)
        jaccardlossesOC_all.extend(_jaccardlossesOC)

    dataframe = pd.DataFrame({'case': names_all,
                              'OD_dice': OD_dice_all, 'OC_dice': OC_dice_all,
                              'OD_jaccard': OD_jaccard_all, 'OC_jaccard': OC_jaccard_all,
                              'dice': dice_all, 'jaccard': jaccard_all, 
                              'vCDR': vCDR_all, 'vCDR_real': vCDR_tar_all, 'vCDRloss': vCDRlosses_all,
                              'dicelossesOD': dicelossesOD_all, 'jaccardlossesOD': jaccardlossesOD_all,
                              'dicelossesOC': dicelossesOC_all, 'jaccardlossesOC': jaccardlossesOC_all,
                              })
    dataframe.to_csv(save_path + "_detail_metrics.csv",
                     index=False, sep=',')
    print('Counting CSV generated!')
    mean_resultframe = pd.DataFrame({
            'OD_dice': mean(OD_dice_all), 'OC_dice': mean(OC_dice_all),
            'OD_jaccard': mean(OD_jaccard_all), 'OC_jaccard': mean(OC_jaccard_all),
            'dice': mean(dice_all), 'jaccard': mean(jaccard_all),
            'vCDR': mean(vCDR_all), 'vCDR_real': mean(vCDR_tar_all), 'vCDRloss': mean(vCDRlosses_all)}, index=[1])
    mean_resultframe.to_csv(save_path + "_mean_metrics.csv", index=0)
    print('Calculating CSV generated!')
if __name__ == "__main__":
    main()

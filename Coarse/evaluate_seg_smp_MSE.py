from albumentations.augmentations.geometric.functional import resize
from segmentation_models_pytorch.utils.losses import MSELoss
import torch
import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from losses import train_DiceLoss_weighted, train_JaccardLoss_weighted
from utils import create_validation_arg_parser, generate_dataset, build_model
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd
import surface_distance
import scipy.spatial
from numpy import mean
import matplotlib.image
from PIL import Image
import cv2
from math import floor

# def caculate_dist(preds_FD):
#     dim = preds_FD.size(0)
#     preds_dist = np.zeros(dim)
#     H=256
#     W=256
#     mask_tmp = preds_FD.detach().cpu().numpy().squeeze()
    
#     Fovea_tmp = mask_tmp[0, :, :]
#     OD_tmp = mask_tmp[1, :, :]
#     [X, Y] = np.where(Fovea_tmp == 1)
#     if X.size == 0:
#         fovea_heart = [H/2, W/2]
#     else:
#         fovea_heart = [(max(X)+min(X)+2)/2, (max(Y)+min(Y)+2)/2]
#     [X, Y] = np.where(OD_tmp == 1)
#     if X.size == 0:
#         OD_heart = [H/2, W/2]
#         OD_R = H*0.05
#     else:
#         OD_heart = [(max(X)+min(X)+2)/2, (max(Y)+min(Y)+2)/2]        
#         OD_R = (max(Y) - min(Y))/2
#     preds_dist = pow(pow(OD_heart[0]-fovea_heart[0], 2) + pow(OD_heart[1]-fovea_heart[1], 2), 0.5) / OD_R
    
#     dist_torch = torch.from_numpy(np.expand_dims(preds_dist, 0))
#     return dist_torch

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


def findcenter(preds_DC):
    # if F
    #
    dim = preds_DC.size(0)
    preds_dist = np.zeros(dim)
    H=256
    W=256
    coordinate_fovea = []
    coordinate_OD = []
    mask_tmp = preds_DC.detach().cpu().numpy().squeeze()
    for i in range(dim):
        Fovea_tmp = mask_tmp[0, :, :]                
        X_foveasum = np.sum(Fovea_tmp, 0).tolist()
        Y_foveasum = np.sum(Fovea_tmp, 1).tolist()
        X_fovea = X_foveasum.index(max(X_foveasum))
        Y_fovea = Y_foveasum.index(max(Y_foveasum))
        coordinate_fovea.append([X_fovea, Y_fovea])
        OD_tmp = mask_tmp[1, :, :]                
        X_ODsum = np.sum(OD_tmp, 0).tolist()
        Y_ODsum = np.sum(OD_tmp, 1).tolist()
        X_OD = X_ODsum.index(max(X_ODsum))
        Y_OD = Y_ODsum.index(max(Y_ODsum))
        coordinate_OD.append([X_OD, Y_OD])
    fovea_torch = torch.from_numpy(np.array(coordinate_fovea))
    OD_torch = torch.from_numpy(np.array(coordinate_OD))
    return fovea_torch, OD_torch

def getvCDR(preds_DC):
    dim = preds_DC.size(0)
    vCDR = []
    mask_tmp = np.round(preds_DC.detach().cpu().numpy().squeeze())
    for i in range(dim):
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

def dist_caculate(coordinate_pre, coordinate_tar):
    coor_pre = coordinate_pre.detach().cpu().numpy().squeeze()
    coor_tar = coordinate_tar.detach().cpu().numpy().squeeze()
    # ((coor_pre[0] - coor_tar[0])^2 + (coor_pre[1] - coor_tar[1])^2)^0.5
    return  pow(pow(coor_pre[0] - coor_tar[0], 2) + pow(coor_pre[1] - coor_tar[1], 2), 0.5)

# def caculate_dist(preds_FD):
#     dim = preds_FD.size(0)
#     preds_dist = np.zeros(dim)

#     mask_tmp = preds_FD.detach().cpu().numpy().squeeze()
#     for i in range(dim):
#         Fovea_tmp = mask_tmp[0, :, :]                
#         X_foveasum = np.sum(Fovea_tmp, 0).tolist()
#         Y_foveasum = np.sum(Fovea_tmp, 1).tolist()
#         X_fovea = X_foveasum.index(min(X_foveasum))
#         Y_fovea = Y_foveasum.index(min(Y_foveasum))
#         # coordinate_fovea.append([X_fovea, Y_fovea])
#         OD_tmp = mask_tmp[1, :, :]                
#         X_ODsum = np.sum(OD_tmp, 0).tolist()
#         Y_ODsum = np.sum(OD_tmp, 1).tolist()
#         X_OD = X_ODsum.index(min(X_ODsum))
#         Y_OD = Y_ODsum.index(min(Y_ODsum))
#         # coordinate_OD.append([X_OD, Y_OD])
#         preds_dist[i] = pow(pow(X_OD-X_fovea, 2) + pow(Y_OD-Y_fovea, 2), 0.5)
#     dist_torch = torch.from_numpy(np.expand_dims(preds_dist, 1))
#     return dist_torch

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

    TP = np.sum(testArray*resultArray)
    FP = np.sum((1-testArray)*resultArray)
    FN = np.sum(testArray*(1-resultArray))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    return precision, recall


def evaluate(model, valLoader, device, save_path):

    model.eval()
    dice_ODo = []
    dice_OCo = []
    jaccard_ODo = []
    jaccard_OCo = []
    preds_vCDRo = []
    tars_vCDRo = []
    MSE_vCDRo = []
    name = []
    F_coordinateo = []
    D_coordinateo = []
    MSE_Fo = []
    MAE_Fo = []
    MSE_Do = []
    MAE_Do = []
    dist_FD_o = []
    torch.set_grad_enabled(False)
    for i, (img_file_name, inputs, input2, mask, targets1, targets2, targets3) in enumerate(tqdm(valLoader)):
         #  img_file_name, image, mask_ODOC, mask_fovea_OD, dist

        output_path_pDC = os.path.join(
            save_path, "pDC_" + os.path.basename(img_file_name[0][0:-4])
        )
        output_path_pFD = os.path.join(
            save_path, "pFD_" + os.path.basename(img_file_name[0][0:-4])
        )

        dice_criterion = train_DiceLoss_weighted(mode='multiclass')
        jaccard_criterion = train_JaccardLoss_weighted(mode='multiclass')
        dist_criterion = smp.utils.losses.MSELoss()
        inputs = inputs.to(device)
        seg_labels = targets1.numpy()
        targets1, targets2, targets3 = targets1.to(device), targets2.to(device), targets3.to(device)
        # print(targets1.shape)
        [_, _, H, W] = targets1.shape

        outputs_DC, outputs_FD, outputs_dist = model(inputs)

        outputs = [outputs_DC, outputs_FD, outputs_dist]

        preds_DC = torch.round(outputs[0])
        preds_FD = outputs[1]
        # DC eval begin
        seg_outputs = preds_DC.detach().cpu().numpy().squeeze()
        seg1_OC = seg_outputs[0].astype('uint8')
        seg1_OC = find_max_region(seg1_OC)
        seg1_OC = 1 - find_max_region(1 - seg1_OC)
        seg1_OC = cv2.resize(seg1_OC, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        seg1_OD = seg_outputs[1].astype('uint8')
        seg1_OD = find_max_region(seg1_OD)
        seg1_OD = 1 - find_max_region(1 - seg1_OD)
        seg1_OD = cv2.resize(seg1_OD, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        seg1_BG = seg_outputs[2].astype('uint8')
        seg1_BG = cv2.resize(seg1_BG, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        preds_DC = np.array([seg1_OC, seg1_OD, seg1_BG])
        # save pred DC
        seg_outputs_save = preds_DC.transpose(1, 2, 0)                                  
        matplotlib.image.imsave(output_path_pDC + 'segDC.bmp', seg_outputs_save)
        preds_DC = torch.from_numpy(np.expand_dims(preds_DC, 0))
        dice_OC, dice_OD = dice_criterion(preds_DC.to(device), targets1.squeeze(1))
        dice_OC = 1 - dice_OC
        dice_OD = 1 - dice_OD
        jaccard_OC, jaccard_OD = jaccard_criterion(preds_DC.to(device), targets1.squeeze(1))
        jaccard_OC = 1 - jaccard_OC
        jaccard_OD = 1 - jaccard_OD
        preds_vCDR = getvCDR(preds_DC)
        tar_vCDR = getvCDR(targets1)
        vCDR_loss = dist_criterion(preds_vCDR, tar_vCDR)
        
        # save crop ODOC part for fine segmentation
        # fundus image\mask\output
        size = 448
        # # #
        # 如果有OD或者OC没有识别出来，会把识别出来的坐标带偏，而且很偏。result2的70.讲道理不应该这么偏
        # # #
        [preds_C_coo, preds_D_coo] = findcenter(preds_DC)
        [a_OD, c_OD] = (preds_D_coo.detach().cpu().numpy().squeeze() + preds_C_coo.detach().cpu().numpy().squeeze())/2 - size/2
        [b_OD, d_OD] = (preds_D_coo.detach().cpu().numpy().squeeze() + preds_C_coo.detach().cpu().numpy().squeeze())/2 + size/2
        a_OD = floor(a_OD)
        b_OD = floor(b_OD)
        c_OD = floor(c_OD)
        d_OD = floor(d_OD)
        if a_OD < 0:
            b_OD -= a_OD
            a_OD = 0
        if c_OD < 0:
            d_OD -= c_OD
            c_OD = 0
        if b_OD > H:
            a_OD -= b_OD - H
            b_OD = H 
        if d_OD > W:
            c_OD -= d_OD - W
            d_OD = W 
        # output crop
        output_path_crop_preDC = output_path_crop_preDC = os.path.join(
            save_path, 'crop' , 'preDC' + str(size) + '/' + os.path.basename(img_file_name[0])
        )
        isExists = os.path.exists(save_path + '/crop/preDC' + str(size))  # fovea 用128会好一些
        if not isExists:
            os.makedirs(save_path + '/crop/preDC' + str(size))
        im_pre = seg_outputs_save[c_OD:d_OD, a_OD:b_OD]*255     # 448 448 3
        cv2.imwrite(output_path_crop_preDC, im_pre)

        #output mask and mask crop
        output_path_coarsemask = os.path.join(
            save_path, 'coarse_mask' + '/' + os.path.basename(img_file_name[0])
        )
        output_path_premask = os.path.join(
            save_path, 'crop' , 'pre_mask' + str(size) + '/' + os.path.basename(img_file_name[0])
        )
        mask_output = np.full([H, W], 255, dtype=np.uint8)
        isExists = os.path.exists(save_path + '/coarse_mask')  # fovea 用128会好一些
        if not isExists:
            os.makedirs(save_path + '/coarse_mask')
            os.makedirs(save_path + '/crop/pre_mask' + str(size))
        mask_output[seg1_OD == 1] = 128
        mask_output[seg1_OC == 1] = 0
        cv2.imwrite(output_path_coarsemask, mask_output)
        pre_mask = mask_output[c_OD:d_OD, a_OD:b_OD]
        cv2.imwrite(output_path_premask, pre_mask)

        # fundu crop
        output_path_crop_fundu = os.path.join(
            save_path, 'crop', 'fundu' + str(size) + '/' + os.path.basename(img_file_name[0])
        )
        isExists = os.path.exists(save_path + '/crop/fundu' + str(size))  # fovea 用128会好一些
        if not isExists:
            os.makedirs(save_path + '/crop/fundu' + str(size))
        im_fundu = input2.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        im_fundu = im_fundu[c_OD:d_OD, a_OD:b_OD]
        cv2.imwrite(output_path_crop_fundu, im_fundu);
        # mask crop
        output_path_crop_mask = os.path.join(
            save_path, 'crop', 'mask' + str(size) + '/' + os.path.basename(img_file_name[0])
        )
        isExists = os.path.exists(save_path + '/crop/mask' + str(size))  # fovea 用128会好一些
        if not isExists:
            os.makedirs(save_path + '/crop/mask' + str(size))
        im_mask = mask.detach().cpu().numpy().squeeze()
        im_mask = im_mask[c_OD:d_OD, a_OD:b_OD]
        cv2.imwrite(output_path_crop_mask, im_mask);
        # FD eval begin here only save OD center and Fovea center and their dist MSE 
        Heatmap_outputs = preds_FD.detach().cpu().numpy().squeeze()
        Heat_F = Heatmap_outputs[0]
        Heat_F = cv2.resize(Heat_F, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
        Heat_D = Heatmap_outputs[1]
        Heat_D = cv2.resize(Heat_D, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
        preds_FD = np.array([Heat_F, Heat_D])
        preds_FD = torch.from_numpy(np.expand_dims(preds_FD, 0))
        # save pred Heatmap FD
        # preds_F = torch.from_numpy(Heat_F) 
        # matplotlib.image.imsave(output_path_pFD + 'HeatF.bmp', preds_F)
        # preds_D = torch.from_numpy(Heat_D) 
        # matplotlib.image.imsave(output_path_pFD + 'HeatD.bmp', preds_D)
        preds_FD_array = Heat_F + Heat_D
        matplotlib.image.imsave(output_path_pFD + 'HeatF.bmp', Heat_F)
        matplotlib.image.imsave(output_path_pFD + 'HeatFD.bmp', preds_FD_array)
        matplotlib.image.imsave(output_path_pFD + 'LocationFD.bmp', np.round(preds_FD_array))
        [preds_F_coo, preds_D_coo] = findcenter(preds_FD)
        [tar_F_coo, tar_D_coo] = findcenter(targets2)
        MSE_F = dist_criterion(preds_F_coo.to(torch.float32), tar_F_coo.to(torch.float32))
        MAE_F = dist_caculate(preds_F_coo, tar_F_coo)
        MSE_D = dist_criterion(preds_D_coo.to(torch.float32), tar_D_coo.to(torch.float32))
        MAE_D = dist_caculate(preds_D_coo, tar_D_coo)
        # dist FD begin and Save FD Dist
        Dist_FD = outputs[2].detach().cpu().numpy().squeeze()
        Dist_FD = cv2.resize(Dist_FD, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
        preds_FD = torch.from_numpy(Dist_FD) 
        matplotlib.image.imsave(output_path_pFD + 'DistFD.bmp', preds_FD)

        # Dist and Dist MSE begin
        Dist_FD_pre = torch.from_numpy(np.expand_dims(Dist_FD, 0))
        # dist_FD_pre = caculate_dist(Dist_FD_pre)
        # dist_FD_tar = caculate_dist(targets3)
        # dist_FD_MSE = dist_criterion(dist_FD_pre.to(device), dist_FD_tar.to(device))
        


        

        name.append(os.path.basename(img_file_name[0]))
        F_coordinateo.append(preds_F_coo.cpu().numpy())
        D_coordinateo.append(preds_D_coo.cpu().numpy())
        MSE_Fo.append(MSE_F.cpu().numpy())
        MSE_Do.append(MSE_D.cpu().numpy())
        MAE_Fo.append(MAE_F)
        MAE_Do.append(MAE_D)
        # jaccard_1o.append(jaccard_F.cpu().numpy())
        # jaccard_2o.append(jaccard_D.cpu().numpy())
        # dist_FD_o.append(dist_FD_MSE.cpu().numpy())
        # ASSD_o.append(ASSD)
        dice_ODo.append(dice_OD.cpu().numpy())
        dice_OCo.append(dice_OC.cpu().numpy())
        jaccard_ODo.append(jaccard_OD.cpu().numpy())
        jaccard_OCo.append(jaccard_OC.cpu().numpy())
        preds_vCDRo.append(preds_vCDR.cpu().numpy().squeeze())
        tars_vCDRo.append(tar_vCDR.cpu().numpy().squeeze())
        MSE_vCDRo.append(vCDR_loss.cpu().numpy())
    # torch.set_grad_enabled(True)
    return name, dice_ODo, dice_OCo, jaccard_ODo, jaccard_OCo, preds_vCDRo, tars_vCDRo, MSE_vCDRo, F_coordinateo, D_coordinateo, MAE_Fo, MAE_Do


def main():
    args = create_validation_arg_parser().parse_args()
    args.loss_type = 'dice'
    args.pretrain = 'imagenet'
    args.distance_type = 'dist_mask'
    args.batch_size = 16  # 64 for unet, 16 for deeplabv3+
    args.LR_seg = 1e-4
    args.num_epochs = 100
    args.aux = False
    args.model_type = 'unet++doublesmp'
    args.encoder = 'timm-resnest50d'
    args.train_path = '/raid5/hehq/MICCAI2021/REFUGE/data'
    args.val_path = '/raid5/hehq/MICCAI2021/REFUGE/data'
    args.save_path = '/raid5/hehq/MICCAI2021/REFUGE/Coarse/_newsave_9fovea=0.05hDist=1/result2'
    args.model_file = '/raid5/hehq/MICCAI2021/REFUGE/Coarse/_newsave_9fovea=0.05hDist=1/best_ODdice_0.8093best_OCdice_0.6519_DCjaccard_0.5845_FMSE_204.8512_DMSE_150.4663_FDdistloss_0.0053_epoch_150.pt'
    args.flod = 1
    args.cuda_no = 0
    _train_path = args.train_path
    _val_path = args.val_path
    # model_file = args.model_file
    # model_file_list = [ # 这次的选择更加偏向MSE的结果，后面应该找到一个相对平均一些的结果
    #     '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save14fovea=0.05hDist=1/unet++doublesmp_imagenet_dice_timm-resnest50d_1/best_ODdice_0.9487best_OCdice_0.8213_DCjaccard_0.8004_FMSE_7.975_DMSE_1.875_FDdistloss_0.0008_epoch_234.pt',
    #     '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save14fovea=0.05hDist=1/unet++doublesmp_imagenet_dice_timm-resnest50d_2/best_ODdice_0.8854best_OCdice_0.7913_DCjaccard_0.7354_FMSE_278.875_DMSE_1993.175_FDdistloss_0.0677_epoch_85.pt',
    #     '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save14fovea=0.05hDist=1/unet++doublesmp_imagenet_dice_timm-resnest50d_3/best_ODdice_0.9315best_OCdice_0.828_DCjaccard_0.7903_FMSE_25.025_DMSE_1358.6_FDdistloss_0.0028_epoch_135.pt',
    #     '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save14fovea=0.05hDist=1/unet++doublesmp_imagenet_dice_timm-resnest50d_4/best_ODdice_0.9424best_OCdice_0.8109_DCjaccard_0.7902_FMSE_603.675_DMSE_959.95_FDdistloss_0.0017_epoch_154.pt',
    #     '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save14fovea=0.05hDist=1/unet++doublesmp_imagenet_dice_timm-resnest50d_5/best_ODdice_0.9518best_OCdice_0.8725_DCjaccard_0.8415_FMSE_2.075_DMSE_1.175_FDdistloss_0.001_epoch_146.pt'
    # ]
    save_path = args.save_path
    flod = args.flod
    distance_type = args.distance_type
    print(distance_type)

    
    namesall = []
    dice_ODoall = []
    dice_OCoall = []
    jaccard_ODoall = []
    jaccard_OCoall = []
    preds_vCDRoall = []
    tars_vCDRoall = []
    MSE_vCDRoall = []
    F_coordinateoall = []
    D_coordinateoall = []
    MAE_Fall = []
    MAE_Dall = []
    # dist_FD_oall = []


    if args.pretrain in ['imagenet', 'ssl', 'swsl', 'instagram']:
        pretrain = args.pretrain
    else:
        pretrain = None

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args.batch_size = 16
    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
    train_file_names = list(range(400))
    val_file_names = list(range(400))
    train_num_list = [str(x).zfill(4) for x in range(1, 401)]
    val_num_list = [str(x).zfill(4) for x in range(401, 801)]
    # for flod in [4, 3, 2, 1, 0]:
    #     if flod == 0:  # flod1
    #         # train_file_names = train_file_names[0:80]
    #         # val_file_names = val_file_names[80:100]
    #         train_tmp_list = list(range(1, 81))
    #         train_num_list = [str(x).zfill(4) for x in train_tmp_list]
    #         val_num_list = [str(x).zfill(4) for x in range(81, 101)]
            
    #     elif flod == 1:  # flod2
    #         # train_file_names = train_file_names[0:60] + train_file_names[80:100]
    #         # val_file_names = val_file_names[60:80]
    #         train_tmp_list = list(range(1, 61)) + list(range(81, 101))
    #         train_num_list = [str(x).zfill(4) for x in train_tmp_list]
    #         val_num_list = [str(x).zfill(4) for x in range(61, 81)]
            
    #     elif flod == 2:  # flod3
    #         # train_file_names = train_file_names[0:40] + train_file_names[60:100]
    #         # val_file_names = val_file_names[40:60]
    #         train_tmp_list = list(range(1, 41)) + list(range(61, 101))
    #         train_num_list = [str(x).zfill(4) for x in train_tmp_list]
    #         val_num_list = [str(x).zfill(4) for x in range(41, 61)]
            
    #     elif flod == 3:  # flod4
    #         # train_file_names = train_file_names[0:20] + train_file_names[40:100]
    #         # val_file_names = val_file_names[20:40]
    #         train_tmp_list = list(range(1, 21)) + list(range(41, 101))
    #         train_num_list = [str(x).zfill(4) for x in train_tmp_list]
    #         val_num_list = [str(x).zfill(4) for x in range(21, 41)]
            
    #     else:  # flod5
    #         # train_file_names = train_file_names[20:100]
    #         # val_file_names = val_file_names[0:20]
    #         train_tmp_list = list(range(21, 101))
    #         train_num_list = [str(x).zfill(4) for x in train_tmp_list]
    #         val_num_list = [str(x).zfill(4) for x in range(1, 21)]
    for x in range(400): train_file_names[x] = args.train_path + '/' + train_num_list[x] + '.bmp'
    for x in range(400): val_file_names[x] = args.train_path + '/' + val_num_list[x] + '.bmp'
    _, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, 1, args.distance_type, False)

    model = build_model(args.model_type, args.encoder, pretrain, aux=False).to(device)
    # model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7
    # model_file = model_file_list[flod]
    model_file = args.model_file
    model.load_state_dict(torch.load(model_file))
    # model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7

    _name, _dice_ODo, _dice_OCo, _jaccard_ODo, _jaccard_OCo, _preds_vCDRo, _tars_vCDRo, _MSE_vCDRo, _F_coordinateo, _D_coordinateo, _MAE_F, _MAE_D = evaluate(model, valid_loader, device, save_path)
    names = []
    dice_ODo = []
    dice_OCo = []
    jaccard_ODo = []
    jaccard_OCo = []
    preds_vCDRo = []
    tars_vCDRo = []
    MSE_vCDRo = []
    F_coordinateo = []
    D_coordinateo = []
    MAE_F = []
    MAE_D = []
    # dist_FD_o = []
    names.extend(_name)
    dice_ODo.extend(_dice_ODo)
    dice_OCo.extend(_dice_OCo)
    jaccard_ODo.extend(_jaccard_ODo)
    jaccard_OCo.extend(_jaccard_OCo)
    preds_vCDRo.extend(_preds_vCDRo)
    tars_vCDRo.extend(_tars_vCDRo)
    MSE_vCDRo.extend(_MSE_vCDRo)
    F_coordinateo.extend(_F_coordinateo)
    D_coordinateo.extend(_D_coordinateo)
    MAE_F.extend(_MAE_F)
    MAE_D.extend(_MAE_D)
    # dist_FD_o.extend(_dist_FD_o)
    # # save for the all data
    # namesall.extend(_name)
    # dice_ODoall.extend(_dice_ODo)
    # dice_OCoall.extend(_dice_OCo)
    # jaccard_ODoall.extend(_jaccard_ODo)
    # jaccard_OCoall.extend(_jaccard_OCo)
    # preds_vCDRoall.extend(_preds_vCDRo)
    # tars_vCDRoall.extend(_tars_vCDRo)
    # MSE_vCDRoall.extend(_MSE_vCDRo)
    # F_coordinateoall.extend(_F_coordinateo)
    # D_coordinateoall.extend(_D_coordinateo)
    # MAE_Fall.extend(_MAE_F)
    # MAE_Dall.extend(_MAE_D)
    # # dist_FD_oall.extend(_dist_FD_o)
    # # name_flag = args.val_path[11:12].replace('/', '_')
    name_flag = flod+1
    print(name_flag)

    dataframe = pd.DataFrame({'case': names, 'MAE_F': MAE_F, 'MAE_D': MAE_D, 'F_coordinate': F_coordinateo, 'D_coordinate': D_coordinateo, 'dice_OD': dice_ODo, 'dice_OC': dice_OCo, 'jaccard_OD': jaccard_ODo, 'jaccard_OC': jaccard_OCo, 'preds_vCDR': preds_vCDRo, 'tars_vCDR': tars_vCDRo, 'MSE_vCDR': MSE_vCDRo})
    dataframe.to_csv(save_path + "/" + str(name_flag) + "_heatmap&seg.csv", index=False, sep=',')
    print('Counting CSV generated!')
    resultframe = pd.DataFrame({'mean_MAE_Fhao': mean(MAE_F), 'MAE_D': mean(MAE_D),'F_coordinate': mean(F_coordinateo), 'D_coordinate': mean(D_coordinateo), 'dice_OD': mean(dice_ODo), 'dice_OC': mean(dice_OCo), 'jaccard_OD': mean(jaccard_ODo), 'jaccard_OC': mean(jaccard_OCo), 'preds_vCDR': mean(preds_vCDRo), 'tars_vCDR': mean(tars_vCDRo), 'MSE_vCDR': mean(MSE_vCDRo)}, index=[1])
    resultframe.to_csv(save_path + "/" + str(name_flag) + "_caculate.csv", index=0)
    print('Calculating CSV generated!')

    # dataframe = pd.DataFrame({'case': namesall, 'MAE_F': MAE_Fall, 'MAE_D': MAE_Dall, 'F_coordinate': F_coordinateoall, 'D_coordinate': D_coordinateoall, 'dice_OD': dice_ODoall, 'dice_OC': dice_OCoall, 'jaccard_OD': jaccard_ODoall, 'jaccard_OC': jaccard_OCoall, 'preds_vCDR': preds_vCDRoall, 'tars_vCDR': tars_vCDRoall, 'MSE_vCDR': MSE_vCDRoall})
    # dataframe.to_csv(save_path + "/" + "_allheatmap&seg.csv", index=False, sep=',')
    # print('Counting CSV generated!')
    # resultframe = pd.DataFrame({'mean_MAE_Fhao': mean(MAE_Fall), 'MAE_D': mean(MAE_Dall),'F_coordinate': mean(F_coordinateoall), 'D_coordinate': mean(D_coordinateoall), 'dice_OD': mean(dice_ODoall), 'dice_OC': mean(dice_OCoall), 'jaccard_OD': mean(jaccard_ODoall), 'jaccard_OC': mean(jaccard_OCoall), 'preds_vCDR': mean(preds_vCDRoall), 'tars_vCDR': mean(tars_vCDRoall), 'MSE_vCDR': mean(MSE_vCDRoall)}, index=[1])
    # resultframe.to_csv(save_path + "/" + "_allcaculate.csv", index=0)
    # print('Calculating CSV generated!')


if __name__ == "__main__":
    main()



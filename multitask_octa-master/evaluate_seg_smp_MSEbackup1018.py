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

def findcenter(preds_FD):
    dim = preds_FD.size(0)
    preds_dist = np.zeros(dim)
    H=256
    W=256
    coordinate_fovea = []
    coordinate_OD = []
    mask_tmp = preds_FD.detach().cpu().numpy().squeeze()
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

def caculate_dist(preds_FD):
    dim = preds_FD.size(0)
    preds_dist = np.zeros(dim)

    mask_tmp = preds_FD.detach().cpu().numpy().squeeze()
    for i in range(dim):
        Fovea_tmp = mask_tmp[0, :, :]                
        X_foveasum = np.sum(Fovea_tmp, 0).tolist()
        Y_foveasum = np.sum(Fovea_tmp, 1).tolist()
        X_fovea = X_foveasum.index(min(X_foveasum))
        Y_fovea = Y_foveasum.index(min(Y_foveasum))
        # coordinate_fovea.append([X_fovea, Y_fovea])
        OD_tmp = mask_tmp[1, :, :]                
        X_ODsum = np.sum(OD_tmp, 0).tolist()
        Y_ODsum = np.sum(OD_tmp, 1).tolist()
        X_OD = X_ODsum.index(min(X_ODsum))
        Y_OD = Y_ODsum.index(min(Y_ODsum))
        # coordinate_OD.append([X_OD, Y_OD])
        preds_dist[i] = pow(pow(X_OD-X_fovea, 2) + pow(Y_OD-Y_fovea, 2), 0.5)
        
    dist_torch = torch.from_numpy(np.expand_dims(preds_dist, 1))
    return dist_torch

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
    name = []
    # dice_1o = []
    # dice_2o = []
    # jaccard_1o = []
    # jaccard_2o = []
    F_coordinateo = []
    D_coordinateo = []
    MSE_Fo = []
    MSE_Do = []
    dist_FD_o = []
    # ASSD_o = []
    # model.eval()
    torch.set_grad_enabled(False)
    for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(tqdm(valLoader)):
         #  img_file_name, image, mask_ODOC, mask_fovea_OD, dist

        output_path_pDC = os.path.join(
            save_path, "pDC_" + os.path.basename(img_file_name[0])
        )
        output_path_pFD = os.path.join(
            save_path, "pFD_" + os.path.basename(img_file_name[0])
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
        outputs_FD_dist = caculate_dist(preds_FD)       # evaluation size[1]  targets.size[1, 1]
        outputs_FD_dist = outputs_FD_dist.to(device)
        # DC eval begin
        seg_outputs = preds_DC.detach().cpu().numpy().squeeze()
        seg1_OC = seg_outputs[0]
        seg1_OC = cv2.resize(seg1_OC, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        seg1_OD = seg_outputs[1]
        seg1_OD = cv2.resize(seg1_OD, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
        seg1_BG = seg_outputs[2]
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
        preds_FD = Heat_F + Heat_D
        matplotlib.image.imsave(output_path_pFD + 'HeatFD.bmp', preds_FD)
        [preds_F_coo, preds_D_coo] = findcenter(preds_DC)
        [tar_F_coo, tar_D_coo] = findcenter(targets2)
        MSE_F = dist_criterion(preds_F_coo.to(torch.float32), tar_F_coo.to(torch.float32))
        MSE_D = dist_criterion(preds_D_coo.to(torch.float32), tar_D_coo.to(torch.float32))
        
        # dist FD begin and Save FD Dist
        Dist_FD = outputs[2].detach().cpu().numpy().squeeze()
        Dist_F = Dist_FD[0]
        Dist_F = cv2.resize(Dist_F, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
        preds_F = torch.from_numpy(Dist_F) 
        matplotlib.image.imsave(output_path_pFD + 'DistF.bmp', preds_F)
        Dist_D = Dist_FD[1]
        Dist_D = cv2.resize(Dist_D, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
        preds_D = torch.from_numpy(Dist_D) 
        matplotlib.image.imsave(output_path_pFD + 'DistD.bmp', preds_D)
        # Dist and Dist MSE begin
        Dist_FD = np.array([Dist_F, Dist_D])
        Dist_FD_pre = torch.from_numpy(np.expand_dims(Dist_FD, 0))
        dist_FD_pre = caculate_dist(Dist_FD_pre)
        dist_FD_tar = caculate_dist(targets3)
        dist_FD_MSE = dist_criterion(dist_FD_pre.to(device), dist_FD_tar.to(device))
        


        # surface_distances = surface_distance.compute_surface_distances(label_seg, predict, spacing_mm=(1, 1))

        # HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        # distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        # distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        # surfel_areas_gt = surface_distances["surfel_areas_gt"]
        # surfel_areas_pred = surface_distances["surfel_areas_pred"]

        # ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) + np.sum(distances_gt_to_pred * surfel_areas_gt))/(np.sum(surfel_areas_gt)+np.sum(surfel_areas_pred))

        # if not os.path.exists(os.path.join(save_path, 'm')):
        #     os.makedirs(os.path.join(save_path, 'm'))
        # if not os.path.exists(os.path.join(save_path, 'o/')):
        #     os.makedirs(os.path.join(save_path, 'o/'))
        

        # cv2.imwrite(output_path_p, (seg_outputs*255.))
        # cv2.imwrite(output_path_m, (preds_DC*255.))
        # cv2.imwrite(output_path_b, (outputs2[1, :, :]*255.))

        name.append(os.path.basename(img_file_name[0]))
        F_coordinateo.append(preds_F_coo.cpu().numpy())
        D_coordinateo.append(preds_D_coo.cpu().numpy())
        MSE_Fo.append(MSE_F.cpu().numpy())
        MSE_Do.append(MSE_D.cpu().numpy())
        # jaccard_1o.append(jaccard_F.cpu().numpy())
        # jaccard_2o.append(jaccard_D.cpu().numpy())
        dist_FD_o.append(dist_FD_MSE.cpu().numpy())
        # ASSD_o.append(ASSD)
        dice_ODo.append(dice_OD.cpu().numpy())
        dice_OCo.append(dice_OC.cpu().numpy())
        jaccard_ODo.append(jaccard_OD.cpu().numpy())
        jaccard_OCo.append(jaccard_OC.cpu().numpy())
    # torch.set_grad_enabled(True)
    return name, dice_ODo, dice_OCo, jaccard_ODo, jaccard_OCo, F_coordinateo, D_coordinateo, MSE_Fo, MSE_Do, dist_FD_o


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
    args.train_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/data'
    args.val_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/data'
    args.save_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save3'
    args.model_file = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save4/unet++doublesmp_imagenet_dice_timm-resnest50d_1/300_DCdice_0.9059_DCjaccard_0.8307_FMSE_1076.05_DMSE_1.225_FDdist_90.8862.pt'
    args.flod = 1
    args.cuda_no = 0
    _train_path = args.train_path
    _val_path = args.val_path
    # model_file = args.model_file
    model_file_list = [
        '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save4/unet++doublesmp_imagenet_dice_timm-resnest50d_1/best_DCdice_0.9195_DCjaccard_0.8529_FMSE_15362.35_DMSE_2.05_FDdist_1748.7977.pt',
        '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save4/unet++doublesmp_imagenet_dice_timm-resnest50d_2/best_DCdice_0.8669_DCjaccard_0.7723_FMSE_14998.475_DMSE_4384.2_FDdist_21550.9219.pt',
        '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save4/unet++doublesmp_imagenet_dice_timm-resnest50d_3/best_DCdice_0.8914_DCjaccard_0.8078_FMSE_22091.1_DMSE_518.625_FDdist_3663.9352_epoch_110.pt',
        '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save4/unet++doublesmp_imagenet_dice_timm-resnest50d_4/300_DCdice_0.8938_DCjaccard_0.8122_FMSE_1899.35_DMSE_8.475_FDdist_80.832.pt',
        '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save4/unet++doublesmp_imagenet_dice_timm-resnest50d_5/best_DCdice_0.9154_DCjaccard_0.847_FMSE_10796.075_DMSE_20256.175_FDdist_27710.3974_epoch_40.pt'
    ]
    save_path = args.save_path
    flod = args.flod
    distance_type = args.distance_type
    print(distance_type)

    
    namesall = []
    dice_ODoall = []
    dice_OCoall = []
    jaccard_ODoall = []
    jaccard_OCoall = []
    F_coordinateoall = []
    D_coordinateoall = []
    MSE_Fall = []
    MSE_Dall = []
    dist_FD_oall = []


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
    train_file_names = list(range(1, 81))
    val_file_names = list(range(1, 21))
    for flod in range(5):
        if flod == 0:  # flod1
            # train_file_names = train_file_names[0:80]
            # val_file_names = val_file_names[80:100]
            train_tmp_list = list(range(1, 81))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            val_num_list = [str(x).zfill(4) for x in range(81, 101)]
            
        elif flod == 1:  # flod2
            # train_file_names = train_file_names[0:60] + train_file_names[80:100]
            # val_file_names = val_file_names[60:80]
            train_tmp_list = list(range(1, 61)) + list(range(81, 101))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            val_num_list = [str(x).zfill(4) for x in range(61, 81)]
            
        elif flod == 2:  # flod3
            # train_file_names = train_file_names[0:40] + train_file_names[60:100]
            # val_file_names = val_file_names[40:60]
            train_tmp_list = list(range(1, 41)) + list(range(61, 101))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            val_num_list = [str(x).zfill(4) for x in range(41, 61)]
            
        elif flod == 3:  # flod4
            # train_file_names = train_file_names[0:20] + train_file_names[40:100]
            # val_file_names = val_file_names[20:40]
            train_tmp_list = list(range(1, 21)) + list(range(41, 101))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            val_num_list = [str(x).zfill(4) for x in range(21, 41)]
            
        else:  # flod5
            # train_file_names = train_file_names[20:100]
            # val_file_names = val_file_names[0:20]
            train_tmp_list = list(range(21, 101))
            train_num_list = [str(x).zfill(4) for x in train_tmp_list]
            val_num_list = [str(x).zfill(4) for x in range(1, 21)]
        for x in range(80): train_file_names[x] = args.train_path + '/' + train_num_list[x] + '.bmp'
        for x in range(20): val_file_names[x] = args.train_path + '/' + val_num_list[x] + '.bmp'
        _, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, 1, args.distance_type, False)

        model = build_model(args.model_type, args.encoder, pretrain, aux=False).to(device)
        # model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7
        model_file = model_file_list[flod]
        model.load_state_dict(torch.load(model_file))
        # model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7

        _name, _dice_ODo, _dice_OCo, _jaccard_ODo, _jaccard_OCo, _F_coordinateo, _D_coordinateo, _MSE_F, _MSE_D, _dist_FD_o = evaluate(model, valid_loader, device, save_path)
        names = []
        dice_ODo = []
        dice_OCo = []
        jaccard_ODo = []
        jaccard_OCo = []
        F_coordinateo = []
        D_coordinateo = []
        MSE_F = []
        MSE_D = []
        dist_FD_o = []
        names.extend(_name)
        dice_ODo.extend(_dice_ODo)
        dice_OCo.extend(_dice_OCo)
        jaccard_ODo.extend(_jaccard_ODo)
        jaccard_OCo.extend(_jaccard_OCo)
        F_coordinateo.extend(_F_coordinateo)
        D_coordinateo.extend(_D_coordinateo)
        MSE_F.extend(_MSE_F)
        MSE_D.extend(_MSE_D)
        dist_FD_o.extend(_dist_FD_o)
        # save for the all data
        namesall.extend(_name)
        dice_ODoall.extend(_dice_ODo)
        dice_OCoall.extend(_dice_OCo)
        jaccard_ODoall.extend(_jaccard_ODo)
        jaccard_OCoall.extend(_jaccard_OCo)
        F_coordinateoall.extend(_F_coordinateo)
        D_coordinateoall.extend(_D_coordinateo)
        MSE_Fall.extend(_MSE_F)
        MSE_Dall.extend(_MSE_D)
        dist_FD_oall.extend(_dist_FD_o)
        # name_flag = args.val_path[11:12].replace('/', '_')
        name_flag = flod+1
        print(name_flag)

        dataframe = pd.DataFrame({'case': names, 'MSE_F': MSE_F, 'MSE_D': MSE_D, 'F_coordinate': F_coordinateo, 'D_coordinate': D_coordinateo, 'dice_OD': dice_ODo, 'dice_OC': dice_OCo, 'jaccard_OD': jaccard_ODo, 'jaccard_OC': jaccard_OCo, 'Dist_FD': dist_FD_o})
        dataframe.to_csv(save_path + "/" + str(name_flag) + "_heatmap&seg.csv", index=False, sep=',')
        print('Counting CSV generated!')
        resultframe = pd.DataFrame({'mean_MSE_Fhao': mean(MSE_F), 'MSE_D': mean(MSE_D),'F_coordinate': mean(F_coordinateo), 'D_coordinate': mean(D_coordinateo), 'dice_OD': mean(dice_ODo), 'dice_OC': mean(dice_OCo), 'jaccard_OD': mean(jaccard_ODo), 'jaccard_OC': mean(jaccard_OCo), 'Dist_FD': mean(dist_FD_o)}, index=[1])
        resultframe.to_csv(save_path + "/" + str(name_flag) + "_caculate.csv", index=0)
        print('Calculating CSV generated!')

    dataframe = pd.DataFrame({'case': namesall, 'MSE_F': MSE_Fall, 'MSE_D': MSE_Dall, 'F_coordinate': F_coordinateoall, 'D_coordinate': D_coordinateoall, 'dice_OD': dice_ODoall, 'dice_OC': dice_OCoall, 'jaccard_OD': jaccard_ODoall, 'jaccard_OC': jaccard_OCoall, 'Dist_FD': dist_FD_oall})
    dataframe.to_csv(save_path + "/" + "_allheatmap&seg.csv", index=False, sep=',')
    print('Counting CSV generated!')
    resultframe = pd.DataFrame({'mean_MSE_Fhao': mean(MSE_Fall), 'MSE_D': mean(MSE_Dall),'F_coordinate': mean(F_coordinateoall), 'D_coordinate': mean(D_coordinateoall), 'dice_OD': mean(dice_ODoall), 'dice_OC': mean(dice_OCoall), 'jaccard_OD': mean(jaccard_ODoall), 'jaccard_OC': mean(jaccard_OCoall), 'Dist_FD': mean(dist_FD_oall)}, index=[1])
    resultframe.to_csv(save_path + "/" + "_allcaculate.csv", index=0)
    print('Calculating CSV generated!')


if __name__ == "__main__":
    main()

from albumentations.augmentations.geometric.functional import resize
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

def caculate_dist(preds_FD):
    dim = preds_FD.size(0)
    preds_dist = np.zeros(dim)
    H=256
    W=256
    mask_tmp = preds_FD.detach().cpu().numpy().squeeze()
    
    Fovea_tmp = mask_tmp[0, :, :]
    OD_tmp = mask_tmp[1, :, :]
    [X, Y] = np.where(Fovea_tmp == 1)
    if X.size == 0:
        fovea_heart = [H/2, W/2]
    else:
        fovea_heart = [(max(X)+min(X)+2)/2, (max(Y)+min(Y)+2)/2]
    [X, Y] = np.where(OD_tmp == 1)
    if X.size == 0:
        OD_heart = [H/2, W/2]
        OD_R = H*0.05
    else:
        OD_heart = [(max(X)+min(X)+2)/2, (max(Y)+min(Y)+2)/2]        
        OD_R = (max(Y) - min(Y))/2
    preds_dist = pow(pow(OD_heart[0]-fovea_heart[0], 2) + pow(OD_heart[1]-fovea_heart[1], 2), 0.5) / OD_R
    
    dist_torch = torch.from_numpy(np.expand_dims(preds_dist, 0))
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
    dice_1o = []
    dice_2o = []
    jaccard_1o = []
    jaccard_2o = []
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

        outputs_DC, outputs_FD = model(inputs)

        outputs = [outputs_DC, outputs_FD]

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
        # FD eval begin
        seg_outputs = preds_FD.detach().cpu().numpy().squeeze()
        seg_F = seg_outputs[0]
        seg_F = cv2.resize(seg_F, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
        seg_D = seg_outputs[1]
        seg_D = cv2.resize(seg_D, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
        seg_BG = seg_outputs[2]
        seg_BG = cv2.resize(seg_BG, (int(W), int(H)), interpolation=cv2.INTER_LINEAR)
        # Fovea&Disc dice
        preds_FD = np.round([seg_F, seg_D, seg_BG])
        seg_outputs_save = preds_FD.transpose(1, 2, 0) 
        # save pred FD 
        preds_FD = torch.from_numpy(preds_FD) 
        matplotlib.image.imsave(output_path_pFD + 'segFD.bmp', seg_outputs_save)
        preds_FD = torch.from_numpy(np.expand_dims(preds_FD, 0))
        dice_F, dice_D = dice_criterion(preds_FD.to(device), targets2.squeeze(1))
        jaccard_F, jaccard_D = jaccard_criterion(preds_FD.to(device), targets2.squeeze(1))
        dice_F = 1 - dice_F
        dice_D = 1 - dice_D
        jaccard_F = 1 - jaccard_F
        jaccard_D = 1 - jaccard_D
        dist_FDloss = dist_criterion(outputs_FD_dist, targets3)
        dist_FDloss = pow(pow(H/256, 2) + pow(W/256, 2), 0.5) * dist_FDloss


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
        dice_1o.append(dice_F.cpu().numpy())
        dice_2o.append(dice_D.cpu().numpy())
        jaccard_1o.append(jaccard_F.cpu().numpy())
        jaccard_2o.append(jaccard_D.cpu().numpy())
        dist_FD_o.append(dist_FDloss.cpu().numpy())
        # ASSD_o.append(ASSD)
        dice_ODo.append(dice_OD.cpu().numpy())
        dice_OCo.append(dice_OC.cpu().numpy())
        jaccard_ODo.append(jaccard_OD.cpu().numpy())
        jaccard_OCo.append(jaccard_OC.cpu().numpy())
    # torch.set_grad_enabled(True)
    return name, dice_1o, dice_2o, jaccard_1o, jaccard_2o, dist_FD_o, dice_ODo, dice_OCo, jaccard_ODo, jaccard_OCo


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
    args.train_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/image'
    args.val_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/image'
    args.save_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/save3'
    args.model_file = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/save3/unet++doublesmp_imagenet_dice_timm-resnest50d_1/best_DCdice_0.8716_DCjaccard_0.7823_FDdice_0.8678_FDjaccard_0.7826.pt'
    args.flod = 1
    args.cuda_no = 0
    _train_path = args.train_path
    _val_path = args.val_path
    model_file = args.model_file
    save_path = args.save_path
    flod = args.flod
    distance_type = args.distance_type
    print(distance_type)

    names = []
    dice_1os = []
    dice_2os = []
    jaccard_1os = []
    jaccard_2os = []
    Dist_FD_os = []
    ASSD_os = []
    dice_ODo = []
    dice_OCo = []
    jaccard_ODo = []
    jaccard_OCo = []
    # for i in range(1, 6):

    #     print(i)

    #     _train_path = args.train_path.replace('XXX', str(i))
    #     _val_path = args.val_path.replace('XXX', str(i))
    #     _model_file = args.model_file.replace('XXX', str(i))

    #     print(_train_path)
    #     print(_val_path)
    #     print(_model_file)

    #     print('Original:', _model_file)
    # file_name = ''
    # acc_max = 0
    # for file in os.listdir(_model_file):
    #     if os.path.isfile(os.path.join(_model_file, file)):
    #         indexs = file.split('_')
    #         if float(indexs[2]) > acc_max:
    #             acc_max = float(indexs[2])
    #             file_name = file

    # print('File name:', file_name)
    # model_file = os.path.join(_model_file, file_name)
    # print('New path:', model_file)

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
    train_file_names = sorted(glob.glob(os.path.join(_train_path, "*.bmp")), key=os.path.getmtime)
    # train_file_names = sorted(train_file_names, key=lambda name: int(name[11:]))
    val_file_names = sorted(glob.glob(os.path.join(_val_path, "*.bmp")), key=os.path.getmtime)
    if flod == 1:  # flod1
        train_file_names = train_file_names[0:80]
        val_file_names = val_file_names[80:100]
        # train_tmp_list = list(range(1, 81))
        # train_num_list = [str(x) for x in train_tmp_list]
        # val_num_list = [str(x) for x in range(81, 101)]
    elif flod == 2:  # flod2
        train_file_names = train_file_names[0:60] + train_file_names[80:100]
        val_file_names = val_file_names[60:80]
        # train_tmp_list = list(range(1, 61)) + list(range(81, 101))
        # train_num_list = [str(x) for x in train_tmp_list]
        # val_num_list = [str(x) for x in range(61, 81)]
    elif flod == 3:  # flod3
        train_file_names = train_file_names[0:40] + train_file_names[60:100]
        val_file_names = val_file_names[40:60]
        # train_tmp_list = list(range(1, 41)) + list(range(61, 101))
        # train_num_list = [str(x) for x in train_tmp_list]
        # val_num_list = [str(x) for x in range(41, 61)]
    elif flod == 4:  # flod4
        train_file_names = train_file_names[0:20] + train_file_names[40:100]
        val_file_names = val_file_names[20:40]
        # train_tmp_list = list(range(1, 21)) + list(range(41, 101))
        # train_num_list = [str(x) for x in train_tmp_list]
        # val_num_list = [str(x) for x in range(21, 41)]
    else:  # flod5
        train_file_names = train_file_names[20:100]
        val_file_names = val_file_names[0:20]
        # train_tmp_list = list(range(21, 101))
        # train_num_list = [str(x) for x in train_tmp_list]
        # val_num_list = [str(x) for x in range(1, 21)]
    _, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, 1, args.distance_type, False)

    model = build_model(args.model_type, args.encoder, pretrain, aux=False).to(device)
    # model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7
    model.load_state_dict(torch.load(model_file))
    # model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7

    _name, _dice_1o, _dice_2o, _jaccard_1o, _jaccard_2o, _dist_FD, _dice_ODo, _dice_OCo, _jaccard_ODo, _jaccard_OCo = evaluate(model, valid_loader, device, save_path)

    names.extend(_name)
    dice_1os.extend(_dice_1o)
    dice_2os.extend(_dice_2o)
    jaccard_1os.extend(_jaccard_1o)
    jaccard_2os.extend(_jaccard_2o)
    Dist_FD_os.extend(_dist_FD)
    # ASSD_os.extend(_ASSD_o)
    dice_ODo.extend(_dice_ODo)
    dice_OCo.extend(_dice_OCo)
    jaccard_ODo.extend(_jaccard_ODo)
    jaccard_OCo.extend(_jaccard_OCo)
    name_flag = args.val_path[11:12].replace('/', '_')
    print(name_flag)

    dataframe = pd.DataFrame({'case': names, 'dice_F': dice_1os, 'dice_D': dice_2os, 'jaccard_F': jaccard_1os, 'jaccard_D': jaccard_2os, 'dice_OD': dice_ODo, 'dice_OC': dice_OCo, 'jaccard_OD': jaccard_ODo, 'jaccard_OC': jaccard_OCo, 'Dist_FD': Dist_FD_os})
    dataframe.to_csv(save_path + "/" + name_flag + "_heatmap&seg.csv", index=False, sep=',')
    print('Counting CSV generated!')
    resultframe = pd.DataFrame({'mean_Dice_Fhao': mean(dice_1os), 'kappa': mean(dice_1os),'recall': mean(dice_1os), 'f1score': mean(dice_1os), 'seg_dice1': mean(dice_1os), 'seg_dice2': mean(dice_2os), 'jaccard1': mean(jaccard_1os), 'jaccard2': mean(jaccard_2os), 'HD': mean(Dist_FD_os), 'ASSD': mean(ASSD_os)}, index=[1])
    resultframe.to_csv(save_path + "/" + name_flag + "_acc_kappa.csv", index=0)
    print('Calculating CSV generated!')
    # with open(os.path.join(save_path, "new_seg_report.txt"), "w") as f:
    #     f.write('Dice 1: ' + str(mean(dice_1os)) + ' + ' + str(std(dice_1os)) + '\r\n')
    #     f.write('Jaccard 1 : ' + str(mean(jaccard_1os)) + ' + ' + str(std(jaccard_1os)) + '\r\n')
    #     f.write('Dice 2 : ' + str(mean(dice_2os)) + ' + ' + str(std(dice_2os)) + '\r\n')
    #     f.write('Jaccard 2 : ' + str(mean(jaccard_2os)) + ' + ' + str(std(jaccard_2os)) + '\r\n')
    #     f.write('HD : ' + str(mean(Dist_FD_os)) + ' + ' + str(std(Dist_FD_os)) + '\r\n')
    #     f.write('ASSD : ' + str(mean(ASSD_os)) + ' + ' + str(std(ASSD_os)) + '\r\n')


if __name__ == "__main__":
    main()

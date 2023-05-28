import torch
import os
import time
import logging
import random
import glob
import segmentation_models_pytorch as smp
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import create_train_arg_parser, build_model, define_loss, AverageMeter, generate_dataset
from losses import *
from sklearn.metrics import cohen_kappa_score

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def findcenter(preds_FD):
    dim = preds_FD.size(0)
    preds_dist = np.zeros(dim)
    H=256
    W=256
    coordinate_fovea = []
    coordinate_OD = []
    mask_tmp = preds_FD.detach().cpu().numpy().squeeze()
    for i in range(dim):
        Fovea_tmp = mask_tmp[i, 0, :, :]                
        X_foveasum = np.sum(Fovea_tmp, 0).tolist()
        Y_foveasum = np.sum(Fovea_tmp, 1).tolist()
        X_fovea = X_foveasum.index(max(X_foveasum))
        Y_fovea = Y_foveasum.index(max(Y_foveasum))
        coordinate_fovea.append([X_fovea, Y_fovea])
        OD_tmp = mask_tmp[i, 1, :, :]                
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
        OC_tmp = mask_tmp[i, 0, :, :]
        OC_tmp = np.round(OC_tmp)
        [X, Y] = np.where(OC_tmp == 1)
        if Y.size == 0:
            Y_OC = 0
        else:
            Y_OC = max(Y) - min(Y)
        
        OD_tmp = mask_tmp[i, 1, :, :]
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



def caculate_dist(preds_FD):
    dim = preds_FD.size(0)
    preds_dist = np.zeros(dim)

    mask_tmp = preds_FD.detach().cpu().numpy().squeeze()
    for i in range(dim):
        Fovea_tmp = mask_tmp[i, 0, :, :]                
        X_foveasum = np.sum(Fovea_tmp, 0).tolist()
        Y_foveasum = np.sum(Fovea_tmp, 1).tolist()
        X_fovea = X_foveasum.index(min(X_foveasum))
        Y_fovea = Y_foveasum.index(min(Y_foveasum))
        # coordinate_fovea.append([X_fovea, Y_fovea])
        OD_tmp = mask_tmp[i, 1, :, :]                
        X_ODsum = np.sum(OD_tmp, 0).tolist()
        Y_ODsum = np.sum(OD_tmp, 1).tolist()
        X_OD = X_ODsum.index(min(X_ODsum))
        Y_OD = Y_ODsum.index(min(Y_ODsum))
        # coordinate_OD.append([X_OD, Y_OD])
        preds_dist[i] = pow(pow(X_OD-X_fovea, 2) + pow(Y_OD-Y_fovea, 2), 0.5)
        
    dist_torch = torch.from_numpy(np.expand_dims(preds_dist, 1))
    return dist_torch


def train(epoch, model, data_loader, optimizer, criterion, device, training=False):
    segDC_losses = AverageMeter("Seg_DC_Loss", ".16f")
    segFD_losses = AverageMeter("Seg_FD_Loss", ".16f")
    dices_OD = AverageMeter("OD_Dice", ".8f")
    dices_OC = AverageMeter("OC_Dice", ".8f")
    jaccards_DC = AverageMeter("dc_Jaccard", ".8f")
    MSEs_F = AverageMeter("MSE_F", ".8f")
    MSEs_D = AverageMeter("MSE_D", ".8f")
    dists_FDloss = AverageMeter("fd_MSE", ".8f")
    dists_FD = AverageMeter("fdDist_MSE", ".8f")
    # clas_losses = AverageMeter("Clas_Loss", ".16f")
    # accs = AverageMeter("Accuracy", ".8f")
    # kappas = AverageMeter("Kappa", ".8f")

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    process = tqdm(data_loader)
    for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(process):
        #   img_file_name, image, mask_ODOC, mask_fovea, dist 
        inputs = inputs.to(device)
        targets1, targets2, targets3 = targets1.to(device), targets2.to(device), targets3.to(device)
        
        targets = [targets1, targets2.squeeze(), targets3.squeeze()]

        if training:
            optimizer.zero_grad()

        outputs_DC, outputs_FD, outputs_dist = model(inputs)     # outputs [32, 3, 512, 512] FD[32, 3]
        # outputs = model(inputs)
        # if not isinstance(outputs, list):
        outputs = [outputs_DC, outputs_FD.squeeze(), outputs_dist.squeeze()]
        preds_DC = torch.round(outputs[0])
        # preds_FD = torch.round(outputs[1])
        # outputs_FD_dist = caculate_dist(preds_FD)
        # outputs_FD_dist = outputs_FD_dist.to(device)
        # DC    
        seg_criterion, dice_criterion, jaccard_criterion, dist_criterion = criterion[0], criterion[1], criterion[2], criterion[3]
        segOC_loss, segOD_loss = seg_criterion(outputs[0], targets[0].to(torch.int64))
        segDC_loss = (segOC_loss + segOD_loss)*0.5
        dice_OC, dice_OD = dice_criterion(preds_DC.squeeze(1), targets[0].squeeze(1))
        dice_OC = 1 - dice_OC
        dice_OD = 1 - dice_OD
        # dice_DC = (dice_OC + dice_OD)*0.5
        jaccard_OC, jaccard_OD = jaccard_criterion(preds_DC.squeeze(1), targets[0].squeeze(1).to(torch.float32))
        jaccard_OC = 1 - jaccard_OC
        jaccard_OD = 1 - jaccard_OD
        jaccard_DC = (jaccard_OC + jaccard_OD)*0.5
        preds_vCDR = getvCDR(preds_DC)
        tar_vCDR = getvCDR(targets[0])
        vCDR_loss = dist_criterion(preds_vCDR, tar_vCDR)
        # FD
        segFD_loss = dist_criterion(outputs[1], targets[1].to(torch.float32))
        # 还差点的MSE差值
        [coo_pre_F, coo_pre_D] = findcenter(outputs[1])
        [tar_F, tar_D] = findcenter(targets[1])
        MSE_F = dist_criterion(coo_pre_F.to(torch.float32), tar_F.to(torch.float32))
        MSE_D = dist_criterion(coo_pre_D.to(torch.float32), tar_D.to(torch.float32))
        tar_FD_dice = torch.round(targets2)
        pre_FD_dice = torch.round(outputs_FD)
        FD_diceloss = dice_criterion(pre_FD_dice, tar_FD_dice)
        # FD dist
        dist_FDloss = dist_criterion(outputs[2], targets[2])
        # dist两层的时候可以加
        # dist_FD_pre = caculate_dist(outputs[2])
        # dist_FD_tar = caculate_dist(targets[2])
        # dist_FD = dist_criterion(dist_FD_pre.to(device), dist_FD_tar.to(device))
        # dist的loss 一般是MSE    


        if training:
            # 3个branch 都有
            if epoch < 50:    
                # total_loss = segDC_loss.to(torch.float32) + FD_diceloss[1].to(torch.float32)
                total_loss = dist_FDloss.to(torch.float32)
            elif epoch < 100:
                # total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32)
                total_loss = dist_FDloss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + FD_diceloss[1].to(torch.float32)
            else:
                # total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + dist_FDloss.to(torch.float32)
                total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + FD_diceloss[1].to(torch.float32) + dist_FDloss.to(torch.float32)
            
            # 没有distance 分支 后面还要看看把其他的分支关掉
            # if epoch < 100:
            #     # total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32)
            #     total_loss = segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + FD_diceloss[1].to(torch.float32)
            # else:
            #     # total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + dist_FDloss.to(torch.float32)
            #     total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + FD_diceloss[1].to(torch.float32)
            
            # # 没有heatmap branch 
            # if epoch < 100:     
            #     # total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32)
            #     total_loss = dist_FDloss.to(torch.float32) 
            # else:
            #     # total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + dist_FDloss.to(torch.float32)
            #     total_loss = segDC_loss.to(torch.float32) + dist_FDloss.to(torch.float32)
            
            # # 没有segmentation branch 
            # if epoch < 100:     
            #     # total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32)
            #     total_loss = dist_FDloss.to(torch.float32) 
            # else:
            #     # total_loss = segDC_loss.to(torch.float32) + segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + dist_FDloss.to(torch.float32)
            #     total_loss = segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + FD_diceloss[1].to(torch.float32) + dist_FDloss.to(torch.float32)
            
            # 只有segmentation branch 
            # total_loss = segDC_loss.to(torch.float32)

            # 只有detection branch 
            # total_loss = segFD_loss.to(torch.float32) + FD_diceloss[0].to(torch.float32) + FD_diceloss[1].to(torch.float32)
            
            # 只有distance measure branch 
            total_loss = dist_FDloss.to(torch.float32)
            total_loss.backward()
        optimizer.step() 

    

        segDC_losses.update(segDC_loss.item(), inputs.size(0))
        dices_OD.update(dice_OD.item(), inputs.size(0))
        dices_OC.update(dice_OC.item(), inputs.size(0))
        jaccards_DC.update(jaccard_DC.item(), inputs.size(0))
        segFD_losses.update(segFD_loss.item(), inputs.size(0))
        MSEs_F.update(MSE_F.item(), inputs.size(0))
        MSEs_D.update(MSE_D.item(), inputs.size(0))
        dists_FDloss.update(dist_FDloss.item(), inputs.size(0))
        # dists_FD.update(dist_FD.item(), inputs.size(0))

        process.set_description('SegDC Loss: ' + str(round(segDC_losses.avg, 4)) + 'SegFD Loss: ' + str(round(segFD_losses.avg, 4)) + ' Dist Loss: ' + str(round(dists_FDloss.avg, 4)))

    epoch_segDC_loss = segDC_losses.avg
    epoch_dice_OD = dices_OD.avg
    epoch_dice_OC = dices_OC.avg
    epoch_jaccard_DC = jaccards_DC.avg
    epoch_segFD_loss = segFD_losses.avg
    epoch_MSE_F = MSEs_F.avg
    epoch_MSE_D = MSEs_D.avg
    epoch_dist_FDloss = dists_FDloss.avg
    epoch_dist_FD = dists_FD.avg
    # epoch_clas_loss = clas_losses.avg
    # epoch_acc = accs.avg
    # epoch_kappa = kappas.avg

    return epoch_segDC_loss, epoch_dice_OD, epoch_dice_OC, epoch_jaccard_DC, epoch_segFD_loss, epoch_MSE_F, epoch_MSE_D, epoch_dist_FDloss, epoch_dist_FD


def main():

    args = create_train_arg_parser().parse_args()
    # args.loss_type = 'dice'
    # args.pretrain = 'imagenet'
    # args.distance_type = 'dist_mask'
    # args.batch_size = 16  # 64 for unet, 16 for deeplabv3+
    # args.LR_seg = 1e-4
    # args.num_epochs = 100
    # args.aux = False
    # args.model_type = 'unet++doublesmp'
    # args.encoder = 'timm-resnest50d'
    # args.train_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/data'
    # args.val_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/data'
    # args.save_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/save'
    # args.flod = 1
    # args.cuda_no = 0
    flod = args.flod
    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    # 设置随机数种子
    setup_seed(20)
    log_path = os.path.join(args.save_path, "summary/")
    writer = SummaryWriter(log_dir=log_path)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = os.path.join(log_path, str(rq) + '.log')
    logging.basicConfig(
        filename=log_name,
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    logging.info(args)
    print(args)
    # file_names = os.listdir(args.train_path)      # 这里读取文件并不是按照顺序来读的，打乱了顺序，但是每次都是按照固定的杂乱顺序来读取的 57 5 46 49 98
    # train_file_names = sorted(glob.glob(os.path.join(args.train_path, "*.bmp")), key=os.path.getmtime)
    # val_file_names = sorted(glob.glob(os.path.join(args.val_path, "*.bmp")), key=os.path.getmtime)
    train_file_names = list(range(1, 81))
    val_file_names = list(range(1, 21))
    if flod == 1:  # flod1
        # train_file_names = train_file_names[0:80]
        # val_file_names = val_file_names[80:100]
        train_tmp_list = list(range(1, 81))
        train_num_list = [str(x).zfill(4) for x in train_tmp_list]
        val_num_list = [str(x).zfill(4) for x in range(81, 101)]
        
    elif flod == 2:  # flod2
        # train_file_names = train_file_names[0:60] + train_file_names[80:100]
        # val_file_names = val_file_names[60:80]
        train_tmp_list = list(range(1, 61)) + list(range(81, 101))
        train_num_list = [str(x).zfill(4) for x in train_tmp_list]
        val_num_list = [str(x).zfill(4) for x in range(61, 81)]
        
    elif flod == 3:  # flod3
        # train_file_names = train_file_names[0:40] + train_file_names[60:100]
        # val_file_names = val_file_names[40:60]
        train_tmp_list = list(range(1, 41)) + list(range(61, 101))
        train_num_list = [str(x).zfill(4) for x in train_tmp_list]
        val_num_list = [str(x).zfill(4) for x in range(41, 61)]
        
    elif flod == 4:  # flod4
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
    logging.info(train_file_names)

    # train_file_names = train_file_names[train_num_list]
    random.shuffle(train_file_names)


    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    if args.pretrain in ['imagenet', 'ssl', 'swsl']:
        pretrain = args.pretrain
        model = build_model(args.model_type, args.encoder, pretrain, aux=False)
    else:
        pretrain = None
        model = build_model(args.model_type, args.encoder, pretrain, aux=False)
    logging.info(model)
    model = model.to(device)
    model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7
    # val batch_size set 4
    train_loader, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, args.batch_size, args.distance_type, args.clahe)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.LR_seg)
    ])

    criterion = [
        define_loss(args.loss_type),
        train_DiceLoss_weighted(mode='multiclass'),
        train_JaccardLoss_weighted(mode='multiclass'),
        smp.utils.losses.MSELoss()
    ]

    max_dice_OD = 0
    max_dice_OC = 0
    min_FD_dist = 10000
    epoch_start = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

        print('\nEpoch: {}'.format(epoch))
      # epoch_segDC_loss, epoch_dice_DC, epoch_jaccard_DC, epoch_segFD_loss, epoch_MSE_F, epoch_MSE_D, epoch_dist_FDloss, epoch_dist_FD
        train_segDC_loss, train_dice_OD, train_dice_OC, train_jaccard_DC, train_segFD_loss, train_MSE_F, train_MSE_D, train_dist_FDloss , train_dist_FD = train(epoch, model, train_loader, optimizer, criterion, device, training=True)
        val_segDC_loss, val_dice_OD, val_dice_OC, val_jaccard_DC, val_segFD_loss, val_MSE_F, val_MSE_D, val_dist_FDloss, val_dist_FD = train(epoch, model, valid_loader, optimizer, criterion, device, training=False)
    # 还差一个dist的loss kappa 变成dist
        epoch_info = "Epoch: {}".format(epoch)
        train_segDC_info = "Training SegDC Loss:    {:.4f}, Training OD_Dice:   {:.4f}, Training OC_Dice:   {:.4f}, Training DC_Jaccard:   {:.4f}".format(train_segDC_loss, train_dice_OD, train_dice_OC, train_jaccard_DC)
        train_segFD_info = "Training SegFD Loss:    {:.4f}, Training F MSE:   {:.4f}, Training D MSE:   {:.4f}, Training dist_FDloss:    {:.4f}".format(train_segFD_loss, train_MSE_F, train_MSE_D, train_dist_FDloss)
        val_segDC_info = "Validation SegDC Loss:  {:.4f}, Validation OD_Dice: {:.4f},, Validation OC_Dice: {:.4f}, Validation DC_Jaccard: {:.4f}".format(val_segDC_loss, val_dice_OD, val_dice_OC, val_jaccard_DC)
        val_segFD_info = "Validation SegFD Loss:  {:.4f}, Validation F MSE: {:.4f}, Validation D MSE: {:.4f}, val_dist_FDloss:    {:.4f}".format(val_segFD_loss, val_MSE_F, val_MSE_D, val_dist_FDloss)
        print(train_segDC_info)
        print(train_segFD_info)
        print(val_segDC_info)
        print(val_segFD_info)
        logging.info(epoch_info)
        logging.info(train_segDC_info)
        logging.info(train_segFD_info)
        logging.info(val_segDC_info)
        logging.info(val_segFD_info)
        writer.add_scalar("train_segDC_loss", train_segDC_loss, epoch)
        writer.add_scalar("train_dice_OD", train_dice_OD, epoch)
        writer.add_scalar("train_dice_OC", train_dice_OC, epoch)
        writer.add_scalar("train_jaccard_DC", train_jaccard_DC, epoch)
        writer.add_scalar("train_segFD_loss", train_segFD_loss, epoch)
        writer.add_scalar("train_MSE_F", train_MSE_F, epoch)
        writer.add_scalar("train_MSE_D", train_MSE_D, epoch)
        writer.add_scalar("train_dist_FDloss", train_dist_FDloss, epoch)
        writer.add_scalar("train_dist_FD", train_dist_FD, epoch)
        writer.add_scalar("val_segDC_loss", val_segDC_loss, epoch)
        writer.add_scalar("val_dice_OD", val_dice_OD, epoch)
        writer.add_scalar("val_dice_OC", val_dice_OC, epoch)
        writer.add_scalar("val_jaccard_DC", val_jaccard_DC, epoch)
        writer.add_scalar("val_seg_lossFD", val_segFD_loss, epoch)
        writer.add_scalar("val_MSE_F", val_MSE_F, epoch)
        writer.add_scalar("val_MSE_D", val_MSE_D, epoch)
        writer.add_scalar("val_dist_FDloss", val_dist_FDloss, epoch)
        writer.add_scalar("val_dist_FD", val_dist_FD, epoch)
        best_name = os.path.join(args.save_path, "best_ODdice_" + str(round(val_dice_OD, 4)) + "best_OCdice_" + str(round(val_dice_OC, 4)) + "_DCjaccard_" + str(round(val_jaccard_DC, 4)) + "_FMSE_" + str(round(val_MSE_F, 4)) + "_DMSE_" + str(round(val_MSE_D, 4)) + "_FDdistloss_" + str(round(val_dist_FDloss, 4)) + "_epoch_" +str(epoch) +  ".pt")
        save_name = os.path.join(args.save_path, str(epoch) + "_ODdice_" + str(round(val_dice_OD, 4)) + "_OCdice_" + str(round(val_dice_OC, 4)) + "_DCjaccard_" + str(round(val_jaccard_DC, 4)) + "_FMSE_" + str(round(val_MSE_F, 4)) + "_DMSE_" + str(round(val_MSE_D, 4)) + "_FDdistloss_" + str(round(val_dist_FDloss, 4)) + ".pt")

        if max_dice_OD < val_dice_OD:
            max_dice_OD = val_dice_OD
            if max_dice_OD > 0.88:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best seg OD model saved!')
                logging.warning('Best seg_OD model saved!')
        if max_dice_OC < val_dice_OC:
            max_dice_OC = val_dice_OC
            if max_dice_OD > 0.83:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best seg OC model saved!')
                logging.warning('Best seg_OC model saved!')
        if min_FD_dist > val_MSE_F:
            min_FD_dist = val_MSE_F
            if min_FD_dist < 1000:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best seg FD model saved!')
                logging.warning('Best seg FD model saved!')
        if epoch % 50 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))
            else:
                torch.save(model.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))


if __name__ == "__main__":
    main()

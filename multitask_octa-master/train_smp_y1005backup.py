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


def caculate_dist(preds_FD):
    dim = preds_FD.size(0)
    preds_dist = np.zeros(dim)
    H=256
    W=256
    mask_tmp = preds_FD.detach().cpu().numpy().squeeze()
    for i in range(dim):
        Fovea_tmp = mask_tmp[i, 0, :, :]
        OD_tmp = mask_tmp[i, 1, :, :]
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
        preds_dist[i] = pow(pow(OD_heart[0]-fovea_heart[0], 2) + pow(OD_heart[1]-fovea_heart[1], 2), 0.5) / OD_R
        
    dist_torch = torch.from_numpy(np.expand_dims(preds_dist, 1))
    return dist_torch


def train(epoch, model, data_loader, optimizer, criterion, device, training=False):
    segDC_losses = AverageMeter("Seg_DC_Loss", ".16f")
    segFD_losses = AverageMeter("Seg_FD_Loss", ".16f")
    dices_DC = AverageMeter("dc_Dice", ".8f")
    jaccards_DC = AverageMeter("dc_Jaccard", ".8f")
    dices_FD = AverageMeter("fd_Dice", ".8f")
    jaccards_FD = AverageMeter("fd_Jaccard", ".8f")
    dists_FD = AverageMeter("fd_MSE", ".8f")
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
        
        targets = [targets1, targets2, targets3]

        if training:
            optimizer.zero_grad()

        outputs_DC, outputs_FD = model(inputs)     # outputs [32, 3, 512, 512] FD[32, 3]
        # outputs = model(inputs)
        # if not isinstance(outputs, list):
        outputs = [outputs_DC, outputs_FD]
        preds_DC = torch.round(outputs[0])
        preds_FD = torch.round(outputs[1])
        outputs_FD_dist = caculate_dist(preds_FD)
        outputs_FD_dist = outputs_FD_dist.to(device)
        
        seg_criterion, dice_criterion, jaccard_criterion, dist_criterion = criterion[0], criterion[1], criterion[2], criterion[3]
        segOC_loss, segOD_loss = seg_criterion(outputs[0], targets[0].to(torch.int64))
        segDC_loss = (segOC_loss + segOD_loss)*0.5
        dice_OC, dice_OD = dice_criterion(preds_DC.squeeze(1), targets[0].squeeze(1))
        dice_OC = 1 - dice_OC
        dice_OD = 1 - dice_OD
        dice_DC = (dice_OC + dice_OD)*0.5
        jaccard_OC, jaccard_OD = jaccard_criterion(preds_DC.squeeze(1), targets[0].squeeze(1))
        jaccard_OC = 1 - jaccard_OC
        jaccard_OD = 1 - jaccard_OD
        jaccard_DC = (jaccard_OC + jaccard_OD)*0.5
        segF_loss, segD_loss = seg_criterion(outputs[1], targets[1].to(torch.int64))
        segFD_loss = (segF_loss + segD_loss)*0.5
        dice_F, dice_D = dice_criterion(preds_FD.squeeze(1), targets[1].squeeze(1))
        dice_F = 1 - dice_F
        dice_D = 1 - dice_D
        dice_FD = (dice_F + dice_D)*0.5
        jaccard_F, jaccard_D = jaccard_criterion(preds_FD.squeeze(1), targets[1].squeeze(1))
        jaccard_F = 1 - jaccard_F
        jaccard_D = 1 - jaccard_D
        jaccard_FD = (jaccard_F + jaccard_D)*0.5
        dist_FDloss = dist_criterion(outputs_FD_dist, targets[2])
        # dist_FDloss = (dist_Floss + dist_Dloss)*0.5
        dist_FD = 1 - dist_FDloss
        # dist的dice 一般是MSE    

        # clf_labels = torch.argmax(targets[3], dim=2).squeeze(1)
        # clf_preds = torch.argmax(label, dim=1)
        # clas_loss = clas_criterion(label, clf_labels)   # CrossEntropy
        # predictions = clf_preds.detach().cpu().numpy()
        # target = clf_labels.detach().cpu().numpy()
        # acc = (predictions == target).sum() / len(predictions)
        # kappa = cohen_kappa_score(predictions, target)

        if training:
            if epoch < 30:
                total_loss = segDC_loss
            elif epoch < 60:
                total_loss = segDC_loss + segFD_loss
            else:
                total_loss = segDC_loss + segFD_loss + dist_FDloss
            total_loss.backward()
        optimizer.step()

        segDC_losses.update(segDC_loss.item(), inputs.size(0))
        dices_DC.update(dice_DC.item(), inputs.size(0))
        jaccards_DC.update(jaccard_DC.item(), inputs.size(0))
        segFD_losses.update(segFD_loss.item(), inputs.size(0))
        dices_FD.update(dice_FD.item(), inputs.size(0))
        jaccards_FD.update(jaccard_FD.item(), inputs.size(0))
        dists_FD.update(dist_FD.item(), inputs.size(0))
        # clas_losses.update(clas_loss.item(), inputs.size(0))
        # accs.update(acc.item(), inputs.size(0))
        # kappas.update(kappa.item(), inputs.size(0))

        process.set_description('SegDC Loss: ' + str(round(segDC_losses.avg, 4)) + 'SegFD Loss: ' + str(round(segFD_losses.avg, 4)) + ' Dist Loss: ' + str(round(dists_FD.avg, 4)))

    epoch_segDC_loss = segDC_losses.avg
    epoch_dice_DC = dices_DC.avg
    epoch_jaccard_DC = jaccards_DC.avg
    epoch_segFD_loss = segFD_losses.avg
    epoch_dice_FD = dices_FD.avg
    epoch_jaccard_FD = jaccards_FD.avg
    epoch_dist_FD = dists_FD.avg
    # epoch_clas_loss = clas_losses.avg
    # epoch_acc = accs.avg
    # epoch_kappa = kappas.avg

    return epoch_segDC_loss, epoch_dice_DC, epoch_jaccard_DC, epoch_segFD_loss, epoch_dice_FD, epoch_jaccard_FD, epoch_dist_FD


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
    # args.train_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/image'
    # args.val_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/image'
    # args.save_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/save'
    # args.flod = 1
    # args.cuda_no = 4
    flod = args.flod
    CUDA_SELECT = "cuda:{}".format(args.cuda_no)

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
    model = torch.nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU  ,int(args.cuda_no)+4,int(args.cuda_no)+5,int(args.cuda_no)+6,int(args.cuda_no)+7
    # val batch_size set 4
    train_loader, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, 4, args.distance_type, args.clahe)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.LR_seg)
    ])

    criterion = [
        define_loss(args.loss_type),
        train_DiceLoss_weighted(mode='multiclass'),
        train_JaccardLoss_weighted(mode='multiclass'),
        smp.utils.losses.MSELoss()
    ]

    max_dice_DC = 0
    max_dice_FD_dist = 0
    epoch_start = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

        print('\nEpoch: {}'.format(epoch))
      # epoch_segDC_loss, epoch_dice_DC, epoch_jaccard_DC, epoch_segFD_loss, epoch_dice_FD, epoch_jaccard_FD, epoch_dist_FD
        train_segDC_loss, train_dice_DC, train_jaccard_DC, train_segFD_loss, train_dice_FD, train_jaccard_FD, train_dist_FD = train(epoch, model, train_loader, optimizer, criterion, device, training=True)
        val_segDC_loss, val_dice_DC, val_jaccard_DC, val_segFD_loss, val_dice_FD, val_jaccard_FD, val_dist_FD = train(epoch, model, valid_loader, optimizer, criterion, device, training=False)
    # 还差一个dist的loss kappa 变成dist
        epoch_info = "Epoch: {}".format(epoch)
        train_segDC_info = "Training SegDC Loss:    {:.4f}, Training DC_Dice:   {:.4f}, Training DC_Jaccard:   {:.4f}".format(train_segDC_loss, train_dice_DC, train_jaccard_DC)
        train_segFD_info = "Training SegFD Loss:    {:.4f}, Training FD_Dice:   {:.4f}, Training FD_Jaccard:   {:.4f}, Training dist_FD:    {:.4f}".format(train_segFD_loss, train_dice_FD, train_jaccard_FD, train_dist_FD)
        val_segDC_info = "Validation SegDC Loss:  {:.4f}, Validation DC_Dice: {:.4f}, Validation DC_Jaccard: {:.4f}".format(val_segDC_loss, val_dice_DC, val_jaccard_DC)
        val_segFD_info = "Validation SegFD Loss:  {:.4f}, Validation FD_Dice: {:.4f}, Validation FD_Jaccard: {:.4f}, val_dist_FD:    {:.4f}".format(val_segFD_loss, val_dice_FD, val_jaccard_FD, val_dist_FD)
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
        writer.add_scalar("train_dice_DC", train_dice_DC, epoch)
        writer.add_scalar("train_jaccard_DC", train_jaccard_DC, epoch)
        writer.add_scalar("train_segFD_loss", train_segFD_loss, epoch)
        writer.add_scalar("train_dice_FD", train_dice_FD, epoch)
        writer.add_scalar("train_jaccard_FD", train_jaccard_FD, epoch)
        writer.add_scalar("train_dist_FD", train_dist_FD, epoch)
        writer.add_scalar("val_segDC_loss", val_segDC_loss, epoch)
        writer.add_scalar("val_dice_DC", val_dice_DC, epoch)
        writer.add_scalar("val_jaccard_DC", val_jaccard_DC, epoch)
        writer.add_scalar("val_seg_lossFD", val_segFD_loss, epoch)
        writer.add_scalar("val_dice_FD", val_dice_FD, epoch)
        writer.add_scalar("val_jaccard_FD", val_jaccard_FD, epoch)
        # writer.add_scalar("val_clas_loss", val_clas_loss, epoch)
        # writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("val_dist_FD", val_dist_FD, epoch)

        best_name = os.path.join(args.save_path, "best_DCdice_" + str(round(val_dice_DC, 4)) + "_DCjaccard_" + str(round(val_jaccard_DC, 4)) + "_FDdice_" + str(round(val_dice_FD, 4)) + "_FDjaccard_" + str(round(val_jaccard_FD, 4)) + ".pt")
        save_name = os.path.join(args.save_path, str(epoch) + "_DCdice_" + str(round(val_dice_DC, 4)) + "_DCjaccard_" + str(round(val_jaccard_DC, 4)) + "_FDdice_" + str(round(val_dice_FD, 4)) + "_FDjaccard_" + str(round(val_jaccard_FD, 4)) + ".pt")

        if max_dice_DC < val_dice_DC:
            max_dice_DC = val_dice_DC
            if max_dice_DC > 0.5:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best seg DC model saved!')
                logging.warning('Best seg_DC model saved!')
        if max_dice_FD_dist < val_dice_FD:
            max_dice_FD_dist = val_dice_FD
            if max_dice_FD_dist > 0.5:
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

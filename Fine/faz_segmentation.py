import torch
import os
import time
import logging
import random
import glob
import segmentation_models_pytorch as smp
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from losses import *
from augmentation import *
import cv2
import albumentations as A
from dataset import *
import math

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


def train(epoch, model, data_loader, optimizer, criterion, device, writer, save_path, training=False):
    ODlosses = AverageMeter("Loss", ".16f")
    OClosses = AverageMeter("Loss", ".16f")
    vCDRlosses = AverageMeter("Loss", ".16f")
    dices = AverageMeter("Dice", ".8f")
    jaccards = AverageMeter("Jaccard", ".8f")
    OD_dices = AverageMeter("Dice", ".8f")
    OD_jaccards = AverageMeter("Jaccard", ".8f")

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    process = tqdm(data_loader)
    for i, (inputs, targets) in enumerate(process):         # 这里的target一直为0，最后只有第三维度的loss有用  因为crop了，只有背景
        # if(len(targets)) <= 1 or (len(inputs)) <= 1 :
        #     continue
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad()

        outputs = model(inputs)
        preds = torch.round(outputs)

        loss_criterion, dice_criterion, jaccard_criterion = criterion[
            0], criterion[1], criterion[2]
        dist_criterion = smp.utils.losses.MSELoss()
        ODloss, loss = loss_criterion(outputs, targets.to(torch.int64))
        # loss = loss_criterion(outputs, targets.to(torch.int64))               # 当classnum为1 的时候要改loss 还要改这里的回复值
        OD_dice, dice = dice_criterion(preds, targets.to(torch.int64))
        # dice = dice_criterion(preds, targets.to(torch.int64))
        OD_dice = 1 - OD_dice                                               # 当classnum多分类时，要加上
        dice = 1 - dice
        # OD_jaccard, jaccard = jaccard_criterion(preds, targets.to(torch.int64))
        # jaccard = jaccard_criterion(preds, targets.to(torch.int64))
        # OD_jaccard = 1 - OD_jaccard
        # jaccard = 1 - jaccard
        dices.update(dice.item(), inputs.size(0))
        # jaccards.update(jaccard.item(), inputs.size(0))
        OD_dices.update(OD_dice.item(), inputs.size(0))                     # 当classnum多分类时，要加上
        # OD_jaccards.update(OD_jaccard.item(), inputs.size(0))
        pre_vCDR = getvCDR(preds)
        tar_vCDR = getvCDR(targets)
        vCDR_loss = dist_criterion(pre_vCDR, tar_vCDR)
        
        OCloss = loss*2 - ODloss
        ODlosses.update(ODloss.item(), inputs.size(0))
        OClosses.update(OCloss.item(), inputs.size(0))
        vCDRlosses.update(vCDR_loss.item(), inputs.size(0))

        if epoch > 100:
            loss = loss + vCDR_loss

        if training:
            loss.backward()
            optimizer.step()

        process.set_description('ODLoss: ' + str(round(ODlosses.avg, 4)) + '  OCLoss: ' + str(round(OClosses.avg, 4)) + '  vCDRLoss: ' + str(round(pow(vCDRlosses.avg, 0.5), 4)))

    epoch_dice = dices.avg
    # epoch_jaccard = jaccards.avg
    epoch_OD_dice = OD_dices.avg                                            # 当classnum多分类时，要加上
    # epoch_OD_jaccard = OD_jaccards.avg
    epoch_vCDR_loss = pow(vCDRlosses.avg, 0.5)

    return epoch_dice, epoch_OD_dice, epoch_vCDR_loss
    # return epoch_dice

    

def main():
    
    # size = 448
    args = create_train_arg_parser().parse_args()
    # args.test_data_path = "/raid5/hehq/MICCAI2021/Test_data/data/"
    # args.test_mask_path = "/raid5/hehq/MICCAI2021/Test_data/mask/"
    # # args.train_data_path = "/raid5/hehq/MICCAI2021/Train_data/aug/aug_img/"
    # # args.train_data_path = "/raid5/hehq/MICCAI2021/Train_data/aug/crop/"+ str(size) +"/crop_img"+ str(size) +"/"
    # args.train_data_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save7fovea=0.05hDist=1/result2/crop/fundu448/'
    # # args.train_data_path = "/raid5/hehq/MICCAI2021/Train_data/aug/crop/OD352/crop_OD352/"
    # # args.train_data_path = "/raid5/hehq/MICCAI2021/Train_data/aug/crop/OD448aug/crop_ODforOC448/"
    # # args.train_mask_path = "/raid5/hehq/MICCAI2021/Train_data/aug/aug_mask/"
    # # args.train_mask_path = "/raid5/hehq/MICCAI2021/Train_data/aug/crop/"+ str(size) +"/crop_mask"+ str(size) +"/"
    # args.train_mask_path = '/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save7fovea=0.05hDist=1/result2/crop/mask448/'
    # # args.train_mask_path = "/raid5/hehq/MICCAI2021/Train_data/aug/crop/OD352/crop_mask_OD352/"
    # # args.train_mask_path = "/raid5/hehq/MICCAI2021/Train_data/aug/crop/OD448aug/crop_mask_ODforOC448/"
    # args.val_data_path = args.train_data_path
    # args.val_mask_path = args.train_mask_path
    # # args.model_type = "unet_smp"
    # args.model_type = "unet++"
    # args.save_path = '/raid5/hehq/MICCAI2021/ISBI/segmentation_fovea_ODOC/model/'
    # # args.encoder = "timm-resnest50d"
    # args.encoder = "timm-efficientnet-b4"
    # args.loss_type = "dice"
    # args.pretrain = "imagenet"
    # args.batch_size = 16
    # args.LR_seg = 4e-4
    # args.classnum = 3
    # args.usenorm = True
    # args.model_file = ""
    # args.encoder_depth = 4
    # args.activation = "sigmoid"
    # args.in_channels = 5
    # args.train_num_list = ""
    # args.test_num_list = ""
    # args.cuda_no = "0"
    # args.target='fineODOC448'
    flod = args.flod
    CUDA_SELECT = "cuda:{}".format(args.cuda_no)
    # log_path = os.path.join(args.save_path, "ZAugfovea/"+ str(flod) +"-EU++/")
    # log_path = os.path.join(args.save_path, "ZAugODcoarse/"+ str(flod) +"-EU++/")              # 跑粗分割的OD，应该还比较好跑，注意159和171
    # log_path = os.path.join(args.save_path, "ZAugODfine/"+ str(size) +"/resize"+ str(flod) +"-EU++/")  # 跑精分割的OD
    log_path = os.path.join(args.save_path, args.target + "/" + str(flod))  # 跑step3精分割的OC
    # log_path = os.path.join(args.save_path, "ZAugOCfinestep3OD/" +"/"+ str(flod) +"-EU++/")  # 跑step3精分割的OC，以OD为标签
    # log_path = os.path.join(args.save_path, "ZAugfovea&OD/"+ str(flod) +"-E&U/")
    # log_path = os.path.join(args.save_path, "ZAugfineODOC/"+ str(size) +"/"+ str(flod) +"-E&U/")
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
    print(args.encoder)

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    # pretrain = args.pretrain
    model = build_model(args.model_type, args.encoder, args.pretrain, args.classnum, args.encoder_depth, args.activation, args.in_channels)
    
    logging.info(model)
    model = model.to(device)
    
    model = nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+1,int(args.cuda_no)+2,int(args.cuda_no)+3]) # multi-GPU
    # model = nn.DataParallel(model,device_ids=[int(args.cuda_no),int(args.cuda_no)+3]) # multi-GPU

    # val_tmp_list = [10, 17, 21, 24, 30, 25, 32, 42, 43, 62, 64, 67, 69, 72, 80, 87, 90, 93, 95, 98]     # fine validation
    # # val_tmp_list = [1, 2, 13, 14, 18, 20, 21, 28, 31, 38, 39, 52, 59, 62, 69, 70, 78, 81, 93, 99]     # coarse validation
    # train_tmp_list = [x for x in range(1, 101) if x not in val_tmp_list]
    # train_num_list = [str(x).zfill(4) + '_' + str(y) for x in train_tmp_list for y in range(1, 31)]
    # val_num_list = [str(x).zfill(4) + '_' + str(y) for x in val_tmp_list for y in range(1, 31)]

   
    if flod == 1: # flod1
        train_tmp_list = list(range(1, 81))
        train_num_list = [str(x).zfill(4)  for x in train_tmp_list ]
        val_num_list = [str(x).zfill(4) for x in range(81, 101) ]
    elif flod == 2: # flod2
        train_tmp_list = list(range(1, 61)) + list(range(81, 101))
        train_num_list = [str(x).zfill(4) for x in train_tmp_list]
        val_num_list = [str(x).zfill(4) for x in range(61, 81)]
    elif flod == 3: # flod3
        train_tmp_list = list(range(1, 41)) + list(range(61, 101))
        train_num_list = [str(x).zfill(4) for x in train_tmp_list]
        val_num_list = [str(x).zfill(4) for x in range(41, 61)]
    elif flod == 4: # flod4
        train_tmp_list = list(range(1, 21)) + list(range(41, 101))
        train_num_list = [str(x).zfill(4) for x in train_tmp_list]
        val_num_list = [str(x).zfill(4) for x in range(21, 41)]
    else: # flod5  
        train_tmp_list = list(range(21, 101))
        train_num_list = [str(x).zfill(4) for x in train_tmp_list]
        val_num_list = [str(x).zfill(4) for x in range(1, 21)]

    m, s = mean_and_std(args.train_data_path, train_num_list)
    print(m, s)

    if args.loss_type == 'dice':
        train_data = TrainDataSet(data_path=args.train_data_path, mask_path=args.train_mask_path,
                                transform=train_crop_aug(m, s), num_list=train_num_list, class_num=args.classnum)
        val_data = TestDataSet(data_path=args.val_data_path, mask_path=args.val_mask_path,
                                transform=val_crop_aug(m, s), num_list=val_num_list, class_num=args.classnum)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=4, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.LR_seg)
    ])

    if args.loss_type == 'dice':
        criterion = [
            weighted_define_loss(args.loss_type, args.classnum),
            train_DiceLoss_weighted(mode='multiclass'),
            train_JaccardLoss_weighted(mode='multiclass')
        ]
    elif args.loss_type == 'ce':
        criterion = [
            weighted_define_loss(args.loss_type, args.classnum),
            train_DiceLoss_logit(mode='multiclass'),
            train_JaccardLoss_logit(mode='multiclass')
        ]
    else :
        criterion = [
            weighted_define_loss(args.loss_type, args.classnum),
            train_DiceLoss_logit(mode='multiclass'),
            train_JaccardLoss_logit(mode='multiclass')
        ]

    max_dice = 0.88 # OC选个小一点的值，方便存好模型
    max_OCdice = 0.8
    max_ODdice = 0.88
    epoch_start = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):  # 500个epoch 人麻了

        print('\nEpoch: {}'.format(epoch))
        if args.classnum == 3:
            train_dice,  train_OD_dice, train_MAE_vCDR = train(
                epoch, model, train_loader, optimizer, criterion, device, writer, args.save_path, training=True)
            val_dice, val_OD_dice, val_MAE_vCDR = train(
                epoch, model, val_loader, optimizer, criterion, device, writer, args.save_path, training=False)
            epoch_info = "Epoch: {}".format(epoch)
            train_info = "Training Dice:{:.4f}, Training OC_Dice:{:.4f},Training OD_Dice:{:.4f}".format(
                train_dice, train_dice*2-train_OD_dice, train_OD_dice)
            val_info = "Val Dice: {:.4f}, Val OC_Dice: {:.4f}, Val OD_Dice: {:.4f}".format(
                val_dice, val_dice*2-val_OD_dice, val_OD_dice)
            print(train_info)
            print(val_info)
            logging.info(epoch_info)
            logging.info(train_info)
            logging.info(val_info)
            writer.add_scalar("train_dice", train_dice, epoch)
            # writer.add_scalar("train_jaccard", train_jaccard, epoch)
            writer.add_scalar("val_dice", val_dice, epoch)
            # writer.add_scalar("val_jaccard", val_jaccard, epoch)
            writer.add_scalar("train_OD_dice", train_OD_dice, epoch)
            # writer.add_scalar("train_OD_jaccard", train_OD_jaccard, epoch)
            writer.add_scalar("val_OD_dice", val_OD_dice, epoch)
            # writer.add_scalar("val_OD_jaccard", val_OD_jaccard, epoch)
            writer.add_scalar("train_OC_dice", train_dice*2-train_OD_dice, epoch)
            writer.add_scalar("val_OC_dice", val_dice*2-val_OD_dice, epoch)
            writer.add_scalar("train_MAE_vCDR", train_MAE_vCDR, epoch)
            writer.add_scalar("val_MAE_vCDR", val_MAE_vCDR, epoch)
            best_name = os.path.join(log_path, "best_dice_" + str(
                round(val_dice, 4)) + "OCDice_" + str(round(val_dice*2-val_OD_dice, 4)) + "ODDice_" + str(round(val_OD_dice, 4)) + "MAE_vCDR_" + str(round(val_MAE_vCDR, 4)) + ".pt")
            save_name = os.path.join(log_path, str(epoch) + "dice_" + str(
                round(val_dice, 4)) + "OCDice_" + str(round(val_dice*2-val_OD_dice, 4)) + "ODDice_" + str(round(val_OD_dice, 4)) + "MAE_vCDR_" + str(round(val_MAE_vCDR, 4)) + ".pt")
        else: # classnum = 1
            train_dice = train(model, train_loader, optimizer, criterion, device, writer, args.save_path, training=True)
            val_dice = train(model, val_loader, optimizer, criterion, device, writer, args.save_path, training=False)
            epoch_info = "Epoch: {}".format(epoch)
            train_info = "Training OC_Dice:{:.4f}".format(train_dice)
            val_info = "Val OC_Dice: {:.4f}".format(val_dice)
            print(train_info)
            print(val_info)
            logging.info(epoch_info)
            logging.info(train_info)
            logging.info(val_info)
            writer.add_scalar("train_dice", train_dice, epoch)
            writer.add_scalar("val_dice", val_dice, epoch)


            best_name = os.path.join(log_path, "best_OCdice_" + str(round(val_dice, 4)) + ".pt")
            save_name = os.path.join(log_path, str(epoch) + "OCdice_" + str(round(val_dice, 4)) + ".pt")
        

        if max_dice < val_dice:
            max_dice = val_dice
            if max_dice > 0.88: # OC 
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best model saved!')
                logging.warning('Best model saved!')
        if max_OCdice < val_dice*2-val_OD_dice:
            max_OCdice = val_dice*2-val_OD_dice
            if max_OCdice > 0.80: # OC 
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best OC model saved!')
                logging.warning('Best model saved!')
        if max_ODdice < val_OD_dice:
            max_ODdice = val_OD_dice
            if max_ODdice > 0.88: # OD
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best OD model saved!')
                logging.warning('Best model saved!')
        if epoch % 50 == 0:
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))
            else:
                torch.save(model.state_dict(), save_name)
                print('Epoch {} model saved!'.format(epoch))


if __name__ == "__main__":
    main()

import torch
import os
import time
import logging
import random
import glob
import segmentation_models_pytorch as smp
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import create_train_arg_parser, build_model, define_loss, AverageMeter, generate_dataset


def train(model, data_loader, optimizer, criterion, device, writer, training=False):
    losses = AverageMeter("Loss", ".16f")
    dices = AverageMeter("Dice", ".8f")
    jaccards = AverageMeter("Jaccard", ".8f")

    if training:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    process = tqdm(data_loader)
    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(process):
        inputs = inputs.to(device)
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        if training:
            optimizer.zero_grad()

        outputs = model(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]
        preds = torch.round(outputs[0])

        loss_criterion, dice_criterion, jaccard_criterion = criterion[0], criterion[1], criterion[2]

        loss = loss_criterion(outputs[0], targets[0].to(torch.float32))
        dice = 1 - dice_criterion(preds.squeeze(1), targets[0].squeeze(1))
        jaccard = 1 - jaccard_criterion(preds.squeeze(1), targets[0].squeeze(1))

        if training:
            loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        dices.update(dice.item(), inputs.size(0))
        jaccards.update(jaccard.item(), inputs.size(0))
        process.set_description('Loss: ' + str(round(losses.avg, 4)))

    epoch_loss = losses.avg
    epoch_dice = dices.avg
    epoch_jaccard = jaccards.avg

    return epoch_loss, epoch_dice, epoch_jaccard


def main():

    args = create_train_arg_parser().parse_args()
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
    print(args.encoder)

    train_file_names = glob.glob(os.path.join(args.train_path, "*.png"))
    random.shuffle(train_file_names)
    val_file_names = glob.glob(os.path.join(args.val_path, "*.png"))

    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    if args.pretrain in ['imagenet', 'ssl', 'swsl']:
        pretrain = args.pretrain
        model = build_model(args.model_type, args.encoder, pretrain)
    else:
        pretrain = None
        model = build_model(args.model_type, args.encoder, pretrain)

    logging.info(model)
    model = model.to(device)

    train_loader, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, args.val_batch_size, args.distance_type, args.clahe)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.LR_seg)
    ])

    criterion = [
        define_loss(args.loss_type),
        smp.utils.losses.DiceLoss(),
        smp.utils.losses.JaccardLoss()
    ]

    max_dice = 0.8
    epoch_start = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + args.num_epochs):

        print('\nEpoch: {}'.format(epoch))

        train_loss, train_dice, train_jaccard = train(model, train_loader, optimizer, criterion, device, writer, training=True)
        val_loss, val_dice, val_jaccard = train(model, valid_loader, optimizer, criterion, device, writer, training=False)

        epoch_info = "Epoch: {}".format(epoch)
        train_info = "Training Loss:   {:.4f}, Training Dice:   {:.4f}, Training Jaccard:   {:.4f}".format(train_loss, train_dice, train_jaccard)
        val_info = "Validation Loss: {:.4f}, Validation Dice: {:.4f}, Validation Jaccard: {:.4f}".format(val_loss, val_dice, val_jaccard)
        print(train_info)
        print(val_info)
        logging.info(epoch_info)
        logging.info(train_info)
        logging.info(val_info)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_dice", train_dice, epoch)
        writer.add_scalar("train_jaccard", train_jaccard, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_dice", val_dice, epoch)
        writer.add_scalar("val_jaccard", val_jaccard, epoch)

        best_name = os.path.join(args.save_path, "best_dice_" + str(round(val_dice, 4)) + "_jaccard_" + str(round(val_jaccard, 4)) + ".pt")
        save_name = os.path.join(args.save_path, str(epoch) + "_dice_" + str(round(val_dice, 4)) + "_jaccard_" + str(round(val_jaccard, 4)) + ".pt")

        if max_dice < val_dice:
            max_dice = val_dice
            if max_dice > 0.8:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_name)
                else:
                    torch.save(model.state_dict(), best_name)
                print('Best model saved!')
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

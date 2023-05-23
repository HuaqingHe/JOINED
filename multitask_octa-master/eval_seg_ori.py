import torch
import os
import glob
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser, generate_dataset, build_model
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd
import surface_distance
import scipy.spatial
from numpy import mean, std


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

    name = []
    dice_1o = []
    dice_2o = []
    jaccard_1o = []
    jaccard_2o = []
    HD_o = []
    ASSD_o = []

    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(tqdm(valLoader)):

        inputs = inputs.to(device)
        seg_labels = targets1.numpy()
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)

        outputs = model(inputs)
        if not isinstance(outputs, list):
            outputs = [outputs]

        seg_outputs = outputs[0].detach().cpu().numpy().squeeze()
        seg_preds = np.round(seg_outputs)

        seg_prs = seg_preds
        dice_1 = f1_score(seg_labels.squeeze(), seg_prs, average='micro')
        dice_2 = getDSC(seg_labels, seg_prs)
        jaccard_1 = jaccard_score(seg_labels.squeeze(), seg_prs, average='micro')
        jaccard_2 = getJaccard(seg_labels, seg_prs)

        label_seg = np.array(seg_labels.squeeze(), dtype=bool)
        predict = np.array(seg_preds, dtype=bool)

        surface_distances = surface_distance.compute_surface_distances(label_seg, predict, spacing_mm=(1, 1))

        HD = surface_distance.compute_robust_hausdorff(surface_distances, 95)

        distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        surfel_areas_gt = surface_distances["surfel_areas_gt"]
        surfel_areas_pred = surface_distances["surfel_areas_pred"]

        ASSD = (np.sum(distances_pred_to_gt * surfel_areas_pred) + np.sum(distances_gt_to_pred * surfel_areas_gt))/(np.sum(surfel_areas_gt)+np.sum(surfel_areas_pred))

        # if not os.path.exists(os.path.join(save_path, 'm')):
        #     os.makedirs(os.path.join(save_path, 'm'))
        # if not os.path.exists(os.path.join(save_path, 'o/')):
        #     os.makedirs(os.path.join(save_path, 'o/'))
        output_path_m = os.path.join(
            save_path, "m_" + os.path.basename(img_file_name[0])
        )
        output_path_p = os.path.join(
            save_path, os.path.basename(img_file_name[0])
        )

        cv2.imwrite(output_path_p, (seg_outputs*255.))
        cv2.imwrite(output_path_m, (seg_preds*255.))
        # cv2.imwrite(output_path_b, (outputs2[1, :, :]*255.))

        name.append(os.path.basename(img_file_name[0]))
        dice_1o.append(dice_1)
        dice_2o.append(dice_2)
        jaccard_1o.append(jaccard_1)
        jaccard_2o.append(jaccard_2)
        HD_o.append(HD)
        ASSD_o.append(ASSD)

    return name, dice_1o, dice_2o, jaccard_1o, jaccard_2o, HD_o, ASSD_o


def main():
    args = create_validation_arg_parser().parse_args()

    names = []
    dice_1os = []
    dice_2os = []
    jaccard_1os = []
    jaccard_2os = []
    HD_os = []
    ASSD_os = []

    for i in range(1, 6):

        print(i)

        _train_path = args.train_path.replace('XXX', str(i))
        _val_path = args.val_path.replace('XXX', str(i))
        _model_file = args.model_file.replace('XXX', str(i))

        print(_train_path)
        print(_val_path)
        print(_model_file)

        print('Original:', _model_file)
        file_name = ''
        acc_max = 0
        for file in os.listdir(_model_file):
            if os.path.isfile(os.path.join(_model_file, file)):
                indexs = file.split('_')
                if float(indexs[2]) > acc_max:
                    acc_max = float(indexs[2])
                    file_name = file

        print('File name:', file_name)
        model_file = os.path.join(_model_file, file_name)
        print('New path:', model_file)
        save_path = args.save_path

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
        train_file_names = glob.glob(os.path.join(_train_path, "*.png"))
        val_file_names = glob.glob(os.path.join(_val_path, "*.png"))

        _, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, 1, args.distance_type, False)
        # train_mean, train_std = mean_and_std(train_file_names)
        # val_file_names = glob.glob(args.val_path)
        # valid_dataset = DatasetImageMaskContourDist(val_file_names, args.distance_type, train_mean, train_std, args.clahe)
        # valid_loader = DataLoader(valid_dataset, num_workers=4)

        model = build_model(args.model_type, args.encoder, pretrain, aux=False).to(device)
        model.load_state_dict(torch.load(model_file))

        _name, _dice_1o, _dice_2o, _jaccard_1o, _jaccard_2o, _HD_o, _ASSD_o = evaluate(model, valid_loader, device, save_path)

        names.extend(_name)
        dice_1os.extend(_dice_1o)
        dice_2os.extend(_dice_2o)
        jaccard_1os.extend(_jaccard_1o)
        jaccard_2os.extend(_jaccard_2o)
        HD_os.extend(_HD_o)
        ASSD_os.extend(_ASSD_o)

    dataframe = pd.DataFrame({'case': names, 'dice1': dice_1os, 'dice2': dice_2os, 'jaccard1': jaccard_1os, 'jaccard2': jaccard_2os, 'HD': HD_os, 'ASSD': ASSD_os})
    dataframe.to_csv(os.path.join(save_path, "new_class.csv"), index=False, sep=',')
    print('Counting CSV generated!')
    resultframe = pd.DataFrame({'seg_dice1': mean(dice_1os), 'seg_dice2': mean(dice_2os), 'jaccard1': mean(jaccard_1os), 'jaccard2': mean(jaccard_2os), 'HD': mean(HD_os), 'ASSD': mean(ASSD_os)}, index=[1])
    resultframe.to_csv(os.path.join(save_path, "new_acc_kappa.csv"), index=0)
    print('Calculating CSV generated!')
    with open(os.path.join(save_path, "new_seg_report.txt"), "w") as f:
        f.write('Dice 1: ' + str(mean(dice_1os)) + ' + ' + str(std(dice_1os)) + '\r\n')
        f.write('Jaccard 1 : ' + str(mean(jaccard_1os)) + ' + ' + str(std(jaccard_1os)) + '\r\n')
        f.write('Dice 2 : ' + str(mean(dice_2os)) + ' + ' + str(std(dice_2os)) + '\r\n')
        f.write('Jaccard 2 : ' + str(mean(jaccard_2os)) + ' + ' + str(std(jaccard_2os)) + '\r\n')
        f.write('HD : ' + str(mean(HD_os)) + ' + ' + str(std(HD_os)) + '\r\n')
        f.write('ASSD : ' + str(mean(ASSD_os)) + ' + ' + str(std(ASSD_os)) + '\r\n')


if __name__ == "__main__":
    main()

import torch
import os
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
from dataset import DatasetImageMaskContourDist
import glob
from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser
import scipy.io as scio
# from collections import OrderedDict

# def cuda(x):
# return x.cuda(async=True) if torch.cuda.is_available() else x


def build_model(model_type):

    if model_type == "unet":
        model = UNet(num_classes=2)
    if model_type == "dcan":
        model = UNet_DCAN(num_classes=2)
    if model_type == "dmtn":
        model = UNet_DMTN(num_classes=2)
    if model_type == "psinet":
        model = PsiNet(num_classes=2)
    if model_type == "convmcd":
        model = UNet_ConvMCD(num_classes=2)

    return model


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()
    val_path = os.path.join(args.val_path, "*.png")
    model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type
    distance_type = args.distance_type
    print(distance_type)

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    val_file_names = glob.glob(val_path)
    valLoader = DataLoader(DatasetImageMaskContourDist(val_file_names, distance_type))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # print(model_type)
    model = build_model(model_type)
    # model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    # myOwnLoad(model)
    # model.load_state_dict(torch.load(model_file),strict=False)
    model.eval()

    for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)

        if model_type == "unet":
            print("unet")
            outputs1 = model(inputs)

            # print(np.array(outputs1).shape)

            outputs1 = np.array([output.detach().cpu().numpy().squeeze() for output in outputs1]).squeeze()

            # print(outputs1.shape)

        elif model_type == "dcan":

            print("dcan")
            outputs1, outputs2 = model(inputs)
            outputs1 = outputs1.detach().cpu().numpy().squeeze()
            outputs2 = outputs2.detach().cpu().numpy().squeeze()

        elif model_type == "dmtn":

            print("dmtn")
            outputs1, outputs2 = model(inputs)
            outputs1 = outputs1.detach().cpu().numpy().squeeze()
            outputs2 = outputs2.detach().cpu().numpy().squeeze()

        else:  # model_type == "psinet" or "convmcd"
            print("psinet or convmcd")
            outputs1, outputs2, outputs3 = model(inputs)
            outputs1 = outputs1.detach().cpu().numpy().squeeze()
            outputs2 = outputs2.detach().cpu().numpy().squeeze()
            outputs3 = outputs3.detach().cpu().numpy().squeeze()
            print(np.array(outputs1).shape)
            print(np.array(outputs2).shape)
            print(np.array(outputs3).shape)

        # elif model_type == "unet":
        res = np.zeros((416, 416))
        indices = np.argmax(outputs1, axis=0)
        res[indices == 1] = 255
        res[indices == 0] = 0
        output_path_m = os.path.join(
            save_path, "m_" + os.path.basename(img_file_name[0])
        )
        output_path_d = os.path.join(
            save_path, "d_" + os.path.basename(img_file_name[0])
        )
        output_path_dmat = os.path.join(
            save_path, os.path.basename(img_file_name[0]).replace('.png', '.mat')
        )
        output_path_p = os.path.join(
            save_path, os.path.basename(img_file_name[0])
        )
        output_path_b = os.path.join(
            save_path, "b_" + os.path.basename(img_file_name[0])
        )

        if model_type == "unet":
            cv2.imwrite(output_path_p, (np.exp(outputs1[1, :, :])*255.))
            cv2.imwrite(output_path_m, res)
        elif model_type == "dcan":
            cv2.imwrite(output_path_p, (np.exp(outputs1[1, :, :])*255.))
            cv2.imwrite(output_path_m, res)
            cv2.imwrite(output_path_b, (np.exp(outputs2[1, :, :])*255.))
        elif model_type == "dmtn":
            cv2.imwrite(output_path_p, (np.exp(outputs1[1, :, :])*255.))
            cv2.imwrite(output_path_m, res)
            cv2.imwrite(output_path_d, outputs2*255.)
            scio.savemat(output_path_dmat, {'dist': outputs2})
        else:  # model_type == "psinet" or "convmcd"
            cv2.imwrite(output_path_p, (np.exp(outputs1[1, :, :])*255.))
            cv2.imwrite(output_path_m, res)
            cv2.imwrite(output_path_b, (np.exp(outputs2[1, :, :])*255.))
            cv2.imwrite(output_path_d, (outputs3*255.))
            scio.savemat(output_path_dmat, {'dist': outputs3})

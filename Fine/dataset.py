import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import cv2
import PIL.Image as Image
import albumentations as A
import torch.nn.functional as F
from torchvision import transforms


class TrainDataSet(Dataset):
    def __init__(self, data_path=None, mask_path=None, transform=None, num_list=None, class_num=None):
        self.transform = transform
        self.data_path = data_path
        self.mask_path = mask_path
        self.num_list = num_list
        self.class_num = class_num

        if self.class_num == 1:
            self.class_values = [0, 255]
        else:
            self.class_values = [0, 128, 255]

    def __getitem__(self, idx):
        # print(self.num_list[idx])
        if self.num_list is not None:
            images = cv2.imread(
                self.data_path + str(self.num_list[idx]) + '.bmp', -1)  # 按照原格式读入数据
            result = cv2.imread(
                self.data_path.replace("fundu", "preDC") + str(self.num_list[idx]) + '.bmp', -1)
            masks = cv2.imread(
                self.mask_path + str(self.num_list[idx]) + '.bmp', 0)
        else:
            images = cv2.imread(
                self.data_path + '.bmp')
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            masks = cv2.imread(
                self.mask_path + '.bmp', 0)
        
        if self.class_num == 1:                     # 分割OC
            # masks = np.where(masks == 192, 1, masks)    # Fovea
            # masks = np.where(masks != 1, 0, masks)
            masks = np.where(masks == 0, 1, masks)    # OC
            masks = np.where(masks != 1, 0, masks)
            # masks = np.where(masks == 128, 0, masks)    # OD
            # masks = np.where(masks == 0, 1, masks)
            # masks = np.where(masks != 1, 0, masks)
        else:
            if self.class_values is not None:
                # mask = [(masks == v) for v in self.class_values]    # 这里有3个value所以会有3层mask   这个是把OD变成一个环
                mask1 = [[masks == 0], [masks <= 128], [masks == 255]]                  # 这个是把OD变成一个实心
                # mask1 = [[masks == 192], [masks <= 128], [masks == 255]]                  # 这个是识别Fovea和OD
            # masks = np.stack(mask, axis=-1).astype('float')
            masks = np.stack(mask1, axis=-1).astype('float').squeeze()         # 把多层mask拼接起来, 还要注意维度问题
            images = np.concatenate((images, result[:, :, 0:2]), axis=2)
            # plt.imshow(masks[:,:,1])
            # plt.save
            # plt.show()


        if self.transform is not None:
            # images = np.pad(images, 8, 'reflect')
            # if self.class_num == 3:
            #     masks = np.pad(masks, ((8, 8), (8, 8), (0, 0)), 'reflect')  # H * W : 2000 2992 —> 2016 3008  这里还不一定是8
            # # elif self.class_num == 2:
            # #     masks = np.pad(masks, ((8, 8), (0, 0)), 'reflect')
            # elif self.class_num == 1:
            #     masks = np.pad(masks, 8, 'reflect')
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
            if self.class_num == 3:             # H W C ——> C H W 
                masks = masks.transpose(0, 2)   # 0 1 2 ——> 2 1 0
                masks = masks.transpose(1, 2)   # 2 1 0 ——> 2 0 1
        return images, masks

    def __len__(self):
        return len(self.num_list)


class TestDataSet(Dataset):
    def __init__(self, data_path=None, mask_path=None, transform=None, num_list=None, class_num=None):
        self.transform = transform
        self.data_path = data_path
        self.mask_path = mask_path
        self.num_list = num_list
        self.class_num = class_num

        if self.class_num == 1:
            self.class_values = [0, 255]
        else:
            self.class_values = [0, 128, 255]

    def __getitem__(self, idx):
        # print(self.num_list[idx])
        if self.num_list is not None:
            images = cv2.imread(
                self.data_path + str(self.num_list[idx]) + '.bmp', -1)
            result = cv2.imread(self.data_path.replace("fundu", "preDC") + str(self.num_list[idx]) + '.bmp', -1)
            masks = cv2.imread(
                self.mask_path + str(self.num_list[idx]) + '.bmp', 0)
        else:
            images = cv2.imread(
                self.data_path + '.bmp')
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            masks = cv2.imread(
                self.mask_path + '.bmp', 0)
                
        if self.class_num == 1:
            # masks = np.where(masks == 192, 1, masks)    # Fovea
            # masks = np.where(masks != 1, 0, masks)
            masks = np.where(masks == 0, 1, masks)    # OC
            masks = np.where(masks != 1, 0, masks)
            # masks = np.where(masks == 128, 0, masks)    # OD
            # masks = np.where(masks == 0, 1, masks)
            # masks = np.where(masks != 1, 0, masks)
        else:
            if self.class_values is not None:
                # mask = [(masks == v) for v in self.class_values]
                mask1 = [[masks == 0], [masks <= 128], [masks == 255]]
                # mask1 = [[masks == 192], [masks <= 128], [masks == 255]]                  # 这个是识别Fovea和OD
            # masks = np.stack(mask, axis=-1).astype('float')
            masks = np.stack(mask1, axis=-1).astype('float').squeeze()
            images = np.concatenate((images, result[:, :, 0:2]), axis=2)
        if self.transform is not None:
            # images = np.pad(images, 8, 'reflect')
            # if self.class_num == 3:
            #     masks = np.pad(masks, ((8, 8), (8, 8), (0, 0)), 'reflect')
            # # elif self.class_num == 2:
            # #     masks = np.pad(masks, ((8, 8), (0, 0)), 'reflect')
            # elif self.class_num == 1:
            #     masks = np.pad(masks, 8, 'reflect')
            transformed = self.transform(image=images, mask=masks)
            images = transformed["image"]
            masks = transformed["mask"]
            if self.class_num == 3:
                masks = masks.transpose(0, 2)
                masks = masks.transpose(1, 2)
        return images, masks

    def __len__(self):
        return len(self.num_list)


class PreDataSet(Dataset):
        def __init__(self, data_path=None, mask_path=None, transform=None, num_list=None, class_num=None):
            self.transform = transform
            self.data_path = data_path
            self.num_list = num_list

        def __getitem__(self, idx):
            if self.num_list is not None:
                images = cv2.imread(
                    self.data_path + str(self.num_list[idx]) + '.bmp', -1)

            else:
                images = cv2.imread(
                    self.data_path + '.bmp')
                images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            [H, W, C] = images.shape
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]

            return images, H, W

        def __len__(self):
            return len(self.num_list)


class CropDataSet(Dataset):
    def __init__(self, data_path=None, mask_path=None, transform=None, num_list=None, class_num=None):
        self.transform = transform
        self.data_path = data_path
        self.mask_path = mask_path
        self.num_list = num_list

    def __getitem__(self, idx):
        if self.num_list is not None:
            images = cv2.imread(
                self.data_path + str(self.num_list[idx]) + '.bmp', -1)
            masks = cv2.imread(
                self.mask_path + str(self.num_list[idx]) + '.bmp', 0)
        else:
            images = cv2.imread(
                self.data_path + '.bmp')
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        [H, W, c] = images.shape        # 不做resize
        if self.transform is not None:
            transformed = self.transform(image=images)
            images = transformed["image"]

        return images, masks, H, W

    def __len__(self):
        return len(self.num_list)

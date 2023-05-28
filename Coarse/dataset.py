from albumentations.core.serialization import load
import torch
import os
import numpy as np
import cv2
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from scipy import io
import matplotlib.image

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


def mean_and_std(paths):
    print('Calculating mean and std of training set for data normalization.')
    m_list, s_list = [], []
    for img_filename in tqdm(paths):
        img = cv2.imread(img_filename)
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    m = m[0][::-1][0]/255
    s = s[0][::-1][0]/255
    print(m)
    print(s)

    return m, s


class DatasetImageMaskContourDist(Dataset):

    # dataset_type(cup,disc,polyp),
    # distance_type(dist_mask,dist_contour,dist_signed)

    def __init__(self, file_names, distance_type, mean, std, clahe):

        self.file_names = file_names
        self.distance_type = distance_type
        self.mean = mean
        self.std = std
        self.clahe = clahe

    def __len__(self):

        return len(self.file_names)

    def __getitem__(self, idx):

        img_file_name = self.file_names[idx]
        image = load_image(img_file_name, self.mean, self.std, self.clahe)
        image2 = load_image_local(img_file_name, self.mean, self.std, self.clahe)
        mask = load_mask_local(img_file_name, self.mean, self.std, self.clahe)
        mask_ODOC = load_mask_ODOC(img_file_name, self.mean, self.std)   # 三分类
        mask_FD = load_mask_FD(img_file_name, self.mean, self.std)
        mask_dist = load_mask_FDdist(img_file_name, self.mean, self.std)    # dist 作为一层
        
        # mask_fovea ,dist= load_mask_foveaOD(img_file_name, self.mean, self.std) # fovea+OD location
        # contour = load_contourheat(img_file_name)
        # dist = load_distance(img_file_name, self.distance_type) # fovea 中心和OD中心的距离
        # cls = load_class(img_file_name)
        
        return img_file_name, image, mask_ODOC, mask_FD, mask_dist 
        # return img_file_name, image, image2, mask, mask_ODOC, mask_FD, mask_dist 
        
        # return img_file_name, image, mask_ODOC, mask_fovea, dist 
        # return image, mask


def clahe_equalized(imgs):
    # print(imgs.shape)
    # assert (len(imgs.shape)==4)  #4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


def load_image(path, mean, std, clahe):

    img = cv2.imread(path, -1)      # 按照原图形式读进来
    # print(type(img))
    # if clahe:
    #     img = clahe_equalized(img)
    #     img = np.array(img ,dtype=np.float32)

    # PIL transform 
    # data_transforms = transforms.Compose(
    #     [
    #         transforms.Resize([512, 512]),
    #         transforms.ToTensor(),
    #         # transforms.Normalize([0.445,], [0.222,]),
    #         transforms.Normalize([mean, ], [std, ]),
    #     ]
    # )
    # img = data_transforms(img)
    size = 256
    process = A.Compose([
        # A.CenterCrop(size, size, 1),
        A.Resize(size, size, 3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    transformd = process(image=img)
    img = transformd["image"]

    return img

def load_image_local(path, mean, std, clahe):
    img = cv2.imread(path, -1)      # 按照原图形式读进来
    process = A.Compose([
        ToTensorV2(),
    ])
    transformd = process(image=img)
    img = transformd["image"]
    return img

def load_mask_local(path, mean, std, clahe):
    masks = cv2.imread(path.replace("data", "mask_DC").replace("bmp", "bmp"), 0)
    process = A.Compose([
        ToTensorV2(),
    ])
    transformd = process(image=masks)
    mask = transformd["image"]
    return mask

def load_mask_ODOC(path, mean, std,):       # 读图片
    # img = cv2.imread(path, -1)
    masks = cv2.imread(path.replace("data", "mask_DC").replace("bmp", "bmp"), 0)
    # make a max area
    OD_tmp = np.array(masks <= 128, dtype='uint8')
    OD_tmp = find_max_region(OD_tmp)
    # make a max area
    OC_tmp = np.array(masks == 0, dtype='uint8')
    OC_tmp = find_max_region(OC_tmp)
    # make a max area
    BG_tmp = np.array(masks == 255, dtype='uint8')
    BG_tmp = find_max_region(BG_tmp)
    mask1 = [OC_tmp, OD_tmp, BG_tmp]
    masks = np.stack(mask1, axis=-1).astype('float').squeeze()  
    # masks = cv2.resize(masks, (512, 512), interpolation=cv2.INTER_NEAREST)
    size = 256
    process = A.Compose([
        # A.CenterCrop(size, size, 1),
        A.Resize(size, size, 3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    transformd = process(image=masks, mask=masks)
    # img = transformd["image"]
    masks = transformd["mask"]
    masks = masks.permute(2, 0, 1)
    # mask[mask == 255] = 1
    # return torch.from_numpy(np.expand_dims(masks, 0)).long()
    return masks

def load_mask_FD(path, mean, std):
    masks_tmp = io.loadmat(path.replace("data", "mask_FD_mat_0.05h").replace("bmp", "mat"))
    masks_F = masks_tmp["mask_F"]
    masks_D = masks_tmp["mask_D"]
    # matplotlib.image.imsave(path.replace("data", "mask_FD_mat_0.05h"), masks_F + masks_D)
    masks = np.array([masks_F, masks_D])
    masks = masks.transpose(1, 2, 0)
    size = 256
    process = A.Compose([
        # A.CenterCrop(size, size, 1),
        A.Resize(size, size, 3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    transformd = process(image=masks, mask=masks)
    masks = transformd["mask"]
    masks = masks.permute(2, 0, 1)
    return masks

def load_mask_FDdist(path, mean, std):
    masks_tmp = io.loadmat(path.replace("data", "mask_FDdist_mat").replace("bmp", "mat"))
    masks = masks_tmp["mask_distFD"]
    # matplotlib.image.imsave(path.replace("data", "mask_FDdist_mat"), masks)
    size = 256
    process = A.Compose([
        # A.CenterCrop(size, size, 1),
        A.Resize(size, size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    transformd = process(image=masks, mask=masks)
    masks = transformd["mask"]
    return masks

def load_mask_dist(path, mean, std):
    masks_tmp = io.loadmat(path.replace("data", "mask_dist_mat1").replace("bmp", "mat"))
    masks_F = masks_tmp["mask_distF"]
    masks_D = masks_tmp["mask_distD"]
    masks = np.array([masks_F, masks_D])
    masks = masks.transpose(1, 2, 0)
    size = 256
    process = A.Compose([
        # A.CenterCrop(size, size, 1),
        A.Resize(size, size, 3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    transformd = process(image=masks, mask=masks)
    masks = transformd["mask"]
    masks = masks.permute(2, 0, 1)
    return masks

def load_mask_foveaOD(path, mean, std,):    
    
    masks = cv2.imread(path.replace("image", "mask").replace("bmp", "bmp"), 0)
    # make a max area
    OD_tmp = np.array(masks <= 128, dtype='uint8')
    OD_tmp = find_max_region(OD_tmp)
    [H, W] = masks.shape
    Fovea_tmp = np.array(masks == 192, dtype='uint8')
    BG_tmp = np.array(masks == 255, dtype='uint8')
    mask1 = [Fovea_tmp, OD_tmp, BG_tmp]
    masks = np.stack(mask1, axis=-1).astype('float').squeeze()  
    # here dist is a relative distance
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
    dist = pow(pow(OD_heart[0]-fovea_heart[0], 2) + pow(OD_heart[1]-fovea_heart[1], 2), 0.5) / OD_R
    size = 256
    process = A.Compose([
        # A.CenterCrop(size, size, 1),
        A.Resize(size, size, 3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    transformd = process(image=masks, mask=masks)
    # img = transformd["image"]
    masks = transformd["mask"]
    masks = masks.permute(2, 0, 1)
    return masks, torch.from_numpy(np.expand_dims(dist, 0)).float()
    # masks = cv2.resize(masks, (512, 512), interpolation=cv2.INTER_NEAREST)
    # return torch.from_numpy(np.expand_dims(masks, 0)).long(), torch.from_numpy(np.expand_dims(dist, 0)).float()

def load_contour(path):

    contour = cv2.imread(path.replace("image", "contour").replace("png", "png"), 0)
    contour[contour == 255] = 1

    return torch.from_numpy(np.expand_dims(contour, 0)).float()


def load_contourheat(path):

    path = path.replace("image", "contour").replace("png", "mat")
    contour = io.loadmat(path)["contour"]

    return torch.from_numpy(np.expand_dims(contour, 0)).float()


def load_class(path):

    cls0 = [1, 0, 0]
    cls1 = [0, 1, 0]
    cls2 = [0, 0, 1]
    if 'N' in os.path.basename(path):
        cls = cls0
    if 'D' in os.path.basename(path):
        cls = cls1
    if 'M' in os.path.basename(path):
        cls = cls2

    return torch.from_numpy(np.expand_dims(cls, 0)).long()


def load_distance(path, distance_type):

    if distance_type == "dist_mask":
        path = path.replace("image", "dis_mask").replace("png", "mat")
        # print (path)
        # print (io.loadmat(path))
        dist = io.loadmat(path)["dis"]

    if distance_type == "dist_contour":
        path = path.replace("image", "dis_contour").replace("png", "mat")
        dist = io.loadmat(path)["c_dis"]

    if distance_type == "dist_signed01":
        path = path.replace("image", "dis_signed01").replace("png", "mat")
        dist = io.loadmat(path)["s_dis01"]

    if distance_type == "dist_signed11":
        path = path.replace("image", "dis_signed11").replace("png", "mat")
        dist = io.loadmat(path)["s_dis11"]

    if distance_type == "dist_fore":
        path = path.replace("image", "dis_fore").replace("png", "mat")
        dist = io.loadmat(path)["f_dis"]

    return torch.from_numpy(np.expand_dims(dist, 0)).float()

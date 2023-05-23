# Multi-task-deep-network
### *Multi-task deep learning based approaches for semantic segmentation in medical images* 

## Dependencies
#### Packages
* *PyTorch*
* *Segmentation_models_pytorch*
* *TensorboardX*
* *OpenCV*
* *numpy*
* *tqdm*

An exhaustive list of packages used could be found in the *requirements.txt* file. Install the same using the following command:

```bash
 conda create --name <env> --file requirements.txt
```

#### Preprocessing
Contour and Distance Maps are pre-computed and can be obtained from binary mask. Sample MATLAB codes can be found here:

* Contour: https://in.mathworks.com/help/images/ref/bwperim.html
* Distance: https://in.mathworks.com/help/images/ref/bwdist.html


#### Directory Structure
Train and Test image folders should contain the following structure:

```
├── contour
    |-- 1.png
    |-- 2.png
    ...
├── dist_contour
    |--1.mat 
    |--2.mat
    ...
├── dist_mask
    |-- 1.mat
    |-- 2.mat
    ...
├── dist_signed
    |-- 1.mat
    |-- 2.mat
    ...
├── image
    |-- 1.jpg
    |-- 2.jpg
    ...
└── mask
    |-- 1.png
    |-- 2.png
    ...
```
## Training code 

To start the training code, you should set the parameters used for training. 

* train_path: Training image path. 
* val_path: Validation image path. 
* save_path: Path for saving results. 
* train_type: Training type, including single classification & segmentation, cotraining or multitask. 
* model_type: Used for single segmentation, cotraining or multitask. The segmentation architecture used. 
* batch_size: Batch size for training stage. 
* val_batch_size: Batch size for validation stage. 
* num_epochs: Total number of epochs for training stage. 
* use_pretrained: Boolean. Use pretrained weights on ImageNet or not. 
* loss_type: Loss for training stage. 
* LR_seg: Learning rate setting for segmentation process. 
* LR_clf: Learning rate setting for classification process. 
* classnum: Used for single classification, cotraining or multitask. Class number for classification. 



For simply start training, you can use our preset shell file named *Demo.sh* with your prepared dataset stored at the local path. Or you can set the parameters listed above to define your own training architecture. The results will be stored at the local path with your dataset name as a folder named as *model_type+loss_type*.  



## Pretrained models

All pretrained models can be found in folder *models*. The models mainly focuses on three datasets: FAZID, OCTAGON and OCTA-500. Further information of the dataset can be found at their official website. 

For each dataset, we prepared four categories of models, including single-task classification, single-task segmentation, cotraining and multitask. 

For single-task classification, the pretrained model includes Resnet50, Resnest50 and Resnext50. 

For single-task segmentation, the pretrained model includes U-Net, U-Net++ and Deeplabv3+. 

The pretrained cotraining model includes permutation and combination of these six architectures. For example, a cotraining model with U-Net architecture and Resnet50 as encoder for multitask verfication. 
pretrain=None
distance_type='dist_signed11'
usenorm=True
visible_device=4
classnum=3
loss='dice'


model_type='unet_smp'
encoder='timm-resnest50d'
train_type='ori'
test_type='ori/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}


model_type='unet_smp'
encoder='timm-resnest50d'
train_type='aug'
test_type='ori/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}


model_type='unet_smp'
encoder='timm-resnest50d'
train_type='ori_crop224'
test_type='ori_crop224/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

model_type='unet_smp'
encoder='timm-resnest50d'
train_type='aug_crop224'
test_type='ori_crop224/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

model_type='unet_smp'
encoder='timm-resnest50d'
train_type='ori_crop256'
test_type='ori_crop256/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

model_type='unet_smp'
encoder='timm-resnest50d'
train_type='aug_crop256'
test_type='ori_crop256/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

pretrain='imagenet'
distance_type='dist_signed11'
usenorm=True
visible_device=4
classnum=3
loss='dice'


model_type='unet_smp'
encoder='timm-resnest50d'
train_type='ori'
test_type='ori/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}


model_type='unet_smp'
encoder='timm-resnest50d'
train_type='aug'
test_type='ori/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}


model_type='unet_smp'
encoder='timm-resnest50d'
train_type='ori_crop224'
test_type='ori_crop224/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

model_type='unet_smp'
encoder='timm-resnest50d'
train_type='aug_crop224'
test_type='ori_crop224/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

model_type='unet_smp'
encoder='timm-resnest50d'
train_type='ori_crop256'
test_type='ori_crop256/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

model_type='unet_smp'
encoder='timm-resnest50d'
train_type='aug_crop256'
test_type='ori_crop256/'

file_output='./results/paper_results/seg/dataset3/'${train_type}'_'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='./dataset3/XXX/'
# base_save_path='./results_seg/XXX/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
base_save_path='./results_seg/dataset3/XXX/'${train_type}'/'${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'/image/'
val_path=${base_path}'test_'${test_type}'image/'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python eval_seg_ori.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}
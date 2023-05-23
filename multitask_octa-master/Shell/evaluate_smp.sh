pretrain='imagenet'
distance_type='dist_signed11'
usenorm=True
visible_device=1
classnum=3
dataset=3
loss='dice'


# model_type='unet_smp'
# encoder='resnet50'
# train_type='aug_crop192/'
# test_type='ori_crop192/'

# file_output='./output_results/dataset'${dataset}'/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

# base_path='../dataset'${dataset}'/XXX/'
# base_save_path='./results'${dataset}'/XXX/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_file=${base_save_path}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

model_type='unet++'
encoder='resnext50_32x4d'
train_type='aug/'
test_type='ori/'

file_output='./output_results/dataset'${dataset}'/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='../dataset'${dataset}'/XXX/'
base_save_path='./results'${dataset}'/XXX/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'image/'
val_path=${base_path}'test_'${test_type}'image'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

# model_type='unet_smp'
# encoder='resnext50_32x4d'
# train_type='aug/'
# test_type='ori/'

# file_output='./output_results/dataset'${dataset}'/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

# base_path='../dataset'${dataset}'/XXX/'
# base_save_path='./results'${dataset}'/XXX/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_file=${base_save_path}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

# model_type='unet++'
# encoder='resnet50'
# train_type='aug/'
# test_type='ori/'

# file_output='./output_results/dataset'${dataset}'/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

# base_path='../dataset'${dataset}'/XXX/'
# base_save_path='./results'${dataset}'/XXX/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_file=${base_save_path}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

# model_type='unet++'
# encoder='timm-resnest50d'
# train_type='aug/'
# test_type='ori/'

# file_output='./output_results/dataset'${dataset}'/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

# base_path='../dataset'${dataset}'/XXX/'
# base_save_path='./results'${dataset}'/XXX/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_file=${base_save_path}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

# model_type='unet++'
# encoder='resnext50_32x4d'
# train_type='aug/'
# test_type='ori/'

# file_output='./output_results/dataset'${dataset}'/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

# base_path='../dataset'${dataset}'/XXX/'
# base_save_path='./results'${dataset}'/XXX/cotrain/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_file=${base_save_path}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}


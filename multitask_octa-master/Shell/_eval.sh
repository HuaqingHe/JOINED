pretrain='imagenet'
distance_type='dist_signed11'
usenorm=True
visible_device=1
classnum=3
loss='dice'


model_type='deeplabv3+'
encoder='resnet50'
train_type='aug_crop192/'
test_type='ori_crop192/'

file_output='./output_results/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'

base_path='../dataset1/1/'
base_save_path='./results/1/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'image/'
val_path=${base_path}'test_'${test_type}'image'
model_name='best_dice_0.9121_jaccard_0.8384_acc_0.7_kappa_0.5438.pt'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

# base_path='../dataset1/2/'
# base_save_path='./results/2/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_name='best_dice_0.8974_jaccard_0.8145_acc_0.7377_kappa_0.5974.pt'
# model_file=${base_save_path}${train_type}${model_name}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

# base_path='../dataset1/3/'
# base_save_path='./results/3/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_name='best_dice_0.8945_jaccard_0.8097_acc_0.7377_kappa_0.5905.pt'
# model_file=${base_save_path}${train_type}${model_name}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

# base_path='../dataset1/4/'
# base_save_path='./results/4/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_name='best_dice_0.9169_jaccard_0.8466_acc_0.3667_kappa_0.0033.pt'
# model_file=${base_save_path}${train_type}${model_name}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

# base_path='../dataset1/5/'
# base_save_path='./results/5/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
# train_path=${base_path}'train_'${train_type}'image/'
# val_path=${base_path}'test_'${test_type}'image'
# model_name='best_dice_0.9169_jaccard_0.8466_acc_0.3667_kappa_0.0033.pt'
# model_file=${base_save_path}${train_type}${model_name}

# CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_smp.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

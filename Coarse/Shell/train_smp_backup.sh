object_type='faz'
loss_type='dice'
pretrain='imagenet'
distance_type='dist_mask'
batch_size=32  # 64 for unet, 16 for deeplabv3+
LR_seg=1e-4
num_epochs=100
<<<<<<< HEAD
visible_device=1
=======
visible_device=3
>>>>>>> 6e1648295f9eacf33cab4f0698f59a5d5c47e87f
aux='False'
dataset=2

# ====================================================================================

<<<<<<< HEAD
model_type='unet++'
encode='resnext50_32x4d'
=======
# model_type='unet_smp'
# encoder='vgg11'
# pretrain='imagenet'
>>>>>>> 6e1648295f9eacf33cab4f0698f59a5d5c47e87f

# batch_size=32
# type='aug'
# test_type='ori'
# size='_crop224/'

<<<<<<< HEAD
batch_size=32
type='ori/'
test_type='ori/'
size=''

train_path='../dataset'${dataset}'/1/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/1/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/1/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}
=======
# train_path='./dataset1/1/train_'${type}${size}'image/'
# val_path='./dataset1/1/test_'${test_type}${size}'image/'
# base_save_path='./results_seg/dataset1/1/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}
>>>>>>> 6e1648295f9eacf33cab4f0698f59a5d5c47e87f

# train_path='./dataset1/2/train_'${type}${size}'image/'
# val_path='./dataset1/2/test_'${test_type}${size}'image/'
# base_save_path='./results_seg/dataset1/2/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='./dataset1/3/train_'${type}${size}'image/'
# val_path='./dataset1/3/test_'${test_type}${size}'image/'
# base_save_path='./results_seg/dataset1/3/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encoder}

train_path='../dataset'${dataset}'/2/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/2/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/2/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset'${dataset}'/3/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/3/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/3/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset'${dataset}'/4/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/4/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/4/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset'${dataset}'/5/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/5/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/5/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}


batch_size=32
type='aug/'
test_type='ori/'
size=''

train_path='../dataset'${dataset}'/1/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/1/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/1/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset'${dataset}'/2/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/2/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/2/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset'${dataset}'/3/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/3/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/3/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset'${dataset}'/4/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/4/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/4/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset'${dataset}'/5/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/5/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/5/cotrain/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}


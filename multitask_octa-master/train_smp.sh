object_type='faz'   #不需要object_type
loss_type='dice'
pretrain='imagenet'
distance_type='dist_mask'
batch_size=16   # 64 for unet, 16 for deeplabv3+
LR_seg=2e-4
num_epochs=300
visible_device=0,1,2,3,4,5,6,7
aux='False'


# ====================================================================================

model_type='unet++doublesmp'
# encode='swin_base_patch4_window7_224'
encode='timm-resnest50d'
train_path='/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/data'
val_path='/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/data'
base_save_path='/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_news1_trans_fovea=0.05hDist=1/'
cuda_no=0

flod=1
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
--cuda_no ${cuda_no} --flod ${flod} \

flod=2
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
--cuda_no ${cuda_no} --flod ${flod} \

flod=3
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
--cuda_no ${cuda_no} --flod ${flod} \

flod=4
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
--cuda_no ${cuda_no} --flod ${flod} \

flod=5
save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
--cuda_no ${cuda_no} --flod ${flod} \

# base_save_path='/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save13fovea=0.05hDist=1/'
# cuda_no=0

# flod=1
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# flod=2
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# flod=3
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# flod=4
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# flod=5
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# base_save_path='/raid5/hehq/MICCAI2021/ISBI/_multitask_octa-master/_save14fovea=0.05hDist=1/'
# cuda_no=0

# flod=1
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# flod=2
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# flod=3
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# flod=4
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \

# flod=5
# save_path=${base_save_path}${model_type}'_'${pretrain}$'_'${loss_type}'_'${encode}'_'${flod}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_y.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux} \
# --cuda_no ${cuda_no} --flod ${flod} \
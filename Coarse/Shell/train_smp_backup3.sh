# Training for single segmentation
# Encoder for unet: vgg11; encoder for unet++: vgg11; encoder for deeplabv3+: xception

object_type='faz'
loss_type='dice'
pretrain='imagenet'
distance_type='dist_mask'
LR_seg=1e-4
num_epochs=100
visible_device=2
aux='False'

# ====================================================================================

model_type='unet++'
encoder='vgg11_bn'


# batch_size=8
# type='ori/'
# test_type='ori/'
# size=''

# train_path='../dataset1/1/train_'${type}${size}'image/'
# val_path='../dataset1/1/test_'${test_type}${size}'image/'
# base_save_path='./results/1/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/2/train_'${type}${size}'image/'
# val_path='../dataset1/2/test_'${test_type}${size}'image/'
# base_save_path='./results/2/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/3/train_'${type}${size}'image/'
# val_path='../dataset1/3/test_'${test_type}${size}'image/'
# base_save_path='./results/3/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/4/train_'${type}${size}'image/'
# val_path='../dataset1/4/test_'${test_type}${size}'image/'
# base_save_path='./results/4/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${Encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/5/train_'${type}${size}'image/'
# val_path='../dataset1/5/test_'${test_type}${size}'image/'
# base_save_path='./results/5/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}


# batch_size=32
# type='ori'
# test_type='ori'
# size='_crop192/'

# train_path='../dataset1/1/train_'${type}${size}'image/'
# val_path='../dataset1/1/test_'${test_type}${size}'image/'
# base_save_path='./results/1/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/2/train_'${type}${size}'image/'
# val_path='../dataset1/2/test_'${test_type}${size}'image/'
# base_save_path='./results/2/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/3/train_'${type}${size}'image/'
# val_path='../dataset1/3/test_'${test_type}${size}'image/'
# base_save_path='./results/3/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/4/train_'${type}${size}'image/'
# val_path='../dataset1/4/test_'${test_type}${size}'image/'
# base_save_path='./results/4/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${Encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/5/train_'${type}${size}'image/'
# val_path='../dataset1/5/test_'${test_type}${size}'image/'
# base_save_path='./results/5/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}


# batch_size=8
# type='aug/'
# test_type='ori/'
# size=''

# train_path='../dataset1/1/train_'${type}${size}'image/'
# val_path='../dataset1/1/test_'${test_type}${size}'image/'
# base_save_path='./results/1/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/2/train_'${type}${size}'image/'
# val_path='../dataset1/2/test_'${test_type}${size}'image/'
# base_save_path='./results/2/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/3/train_'${type}${size}'image/'
# val_path='../dataset1/3/test_'${test_type}${size}'image/'
# base_save_path='./results/3/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/4/train_'${type}${size}'image/'
# val_path='../dataset1/4/test_'${test_type}${size}'image/'
# base_save_path='./results/4/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${Encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

# train_path='../dataset1/5/train_'${type}${size}'image/'
# val_path='../dataset1/5/test_'${test_type}${size}'image/'
# base_save_path='./results/5/seg/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}


batch_size=32
type='aug'
test_type='ori'
size='_crop192/'

train_path='../dataset1/1/train_'${type}${size}'image/'
val_path='../dataset1/1/test_'${test_type}${size}'image/'
base_save_path='./results/1/seg/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset1/2/train_'${type}${size}'image/'
val_path='../dataset1/2/test_'${test_type}${size}'image/'
base_save_path='./results/2/seg/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset1/3/train_'${type}${size}'image/'
val_path='../dataset1/3/test_'${test_type}${size}'image/'
base_save_path='./results/3/seg/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset1/4/train_'${type}${size}'image/'
val_path='../dataset1/4/test_'${test_type}${size}'image/'
base_save_path='./results/4/seg/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

train_path='../dataset1/5/train_'${type}${size}'image/'
val_path='../dataset1/5/test_'${test_type}${size}'image/'
base_save_path='./results/5/seg/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}'_'${loss_type}'_'${encoder}

CUDA_VISIBLE_DEVICES=${visible_device} python train_smp_seg.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --aux ${aux}

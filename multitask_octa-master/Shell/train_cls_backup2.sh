object_type='faz'
loss_type='dice'
pretrain='True'
distance_type='dist_mask'
LR_seg=1e-4
num_epochs=100
visible_device=3
aux='False'
encode='vgg16'
classnum=2
dataset=2

# ====================================================================================

model_type='resnest50'
pretrain='True'


batch_size=32
type='aug/'
test_type='ori/'
size=''

# train_path='../dataset'${dataset}'/1/train_'${type}${size}'image/'
# val_path='../dataset'${dataset}'/1/test_'${test_type}${size}'image/'
# base_save_path='./results'${dataset}'/1/cls/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}

# train_path='../dataset'${dataset}'/2/train_'${type}${size}'image/'
# val_path='../dataset'${dataset}'/2/test_'${test_type}${size}'image/'
# base_save_path='./results'${dataset}'/2/cls/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}

train_path='../dataset'${dataset}'/3/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/3/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/3/cls/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}

CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}

# train_path='../dataset'${dataset}'/4/train_'${type}${size}'image/'
# val_path='../dataset'${dataset}'/4/test_'${test_type}${size}'image/'
# base_save_path='./results'${dataset}'/4/cls/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}

# train_path='../dataset'${dataset}'/5/train_'${type}${size}'image/'
# val_path='../dataset'${dataset}'/5/test_'${test_type}${size}'image/'
# base_save_path='./results'${dataset}'/5/cls/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}


model_type='resnest50'
pretrain='False'


batch_size=32
type='aug/'
test_type='ori/'
size=''

# train_path='../dataset'${dataset}'/1/train_'${type}${size}'image/'
# val_path='../dataset'${dataset}'/1/test_'${test_type}${size}'image/'
# base_save_path='./results'${dataset}'/1/cls/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}

# train_path='../dataset'${dataset}'/2/train_'${type}${size}'image/'
# val_path='../dataset'${dataset}'/2/test_'${test_type}${size}'image/'
# base_save_path='./results'${dataset}'/2/cls/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}

train_path='../dataset'${dataset}'/3/train_'${type}${size}'image/'
val_path='../dataset'${dataset}'/3/test_'${test_type}${size}'image/'
base_save_path='./results'${dataset}'/3/cls/'${type}${size}
save_path=${base_save_path}${model_type}'_'${pretrain}

CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}

# train_path='../dataset'${dataset}'/4/train_'${type}${size}'image/'
# val_path='../dataset'${dataset}'/4/test_'${test_type}${size}'image/'
# base_save_path='./results'${dataset}'/4/cls/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}

# train_path='../dataset'${dataset}'/5/train_'${type}${size}'image/'
# val_path='../dataset'${dataset}'/5/test_'${test_type}${size}'image/'
# base_save_path='./results'${dataset}'/5/cls/'${type}${size}
# save_path=${base_save_path}${model_type}'_'${pretrain}

# CUDA_VISIBLE_DEVICES=${visible_device} python train_cls.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encode ${encode} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --num_epochs ${num_epochs} --classnum ${classnum}


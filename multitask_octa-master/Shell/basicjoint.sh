model_type='cotrain'
object_type='faz'
loss_type='dice'
distance_type='dist_signed11'


LR_seg=1e-4
LR_clf=1e-4
num_epochs=150  ##

usenorm=True

base_path='./dataset3/1/'
base_save_path='./results/dataset3/1/model/'

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain='imagenet'
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain=None
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3


NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

base_path='./dataset3/2/'
base_save_path='./results/dataset3/2/model/'

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain='imagenet'
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain=None
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3


NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

base_path='./dataset3/3/'
base_save_path='./results/dataset3/3/model/'

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain='imagenet'
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain=None
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3


NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

base_path='./dataset3/4/'
base_save_path='./results/dataset3/4/model/'

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain='imagenet'
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain=None
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3


NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

base_path='./dataset3/5/'
base_save_path='./results/dataset3/5/model/'

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain='imagenet'
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
encoder='timm-resnest50d'
pretrain=None
batch_size=32  ##
visible_device=7  ##
startpoint=10  ##
save_path=${base_save_path}'bascijoint_dice_traug_0212_'${pretrain}'_'${encoder}
classnum=3


NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python basic_jointlearning.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum}
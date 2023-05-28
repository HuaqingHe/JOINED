model_type='cotrain'
object_type='faz'
loss_type='dice'
distance_type='dist_signed11'

LR_seg=1e-4
LR_clf=1e-4
num_epochs=300  ##
usenorm=True

base_path='./dataset3/1/'
base_save_path='./results/dataset3/1/model/'

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
batch_size=32  ##
visible_device=1  ##
startpoint=10  ##
encoder='timm-skresnext50_32x4d' #timm-skresnext50_32x4d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='timm-resnest50d' #timm-skresnext50_32x4d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='timm-resnest50d_4s2x40d' #timm-skresnext50_32x4d  timm-resnest50d_4s2x40d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='timm-resnest26d' #timm-skresnext50_32x4d  timm-resnest50d_4s2x40d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='resnext50_32x4d' #timm-skresnext50_32x4d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='resnet50' #timm-skresnext50_32x4d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

base_path='./dataset3/2/'
base_save_path='./results/dataset3/2/model/'

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
batch_size=32  ##
visible_device=1  ##
startpoint=10  ##
encoder='timm-skresnext50_32x4d' #timm-skresnext50_32x4d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='timm-resnest50d' #timm-skresnext50_32x4d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='timm-resnest50d_4s2x40d' #timm-skresnext50_32x4d  timm-resnest50d_4s2x40d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='timm-resnest26d' #timm-skresnext50_32x4d  timm-resnest50d_4s2x40d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='resnext50_32x4d' #timm-skresnext50_32x4d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

encoder='resnet50' #timm-skresnext50_32x4d
attention=None
save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
classnum=3
pretrain='imagenet'

NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
--encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 


# base_path='./dataset3/3/'
# base_save_path='./results/dataset3/3/model/'

# train_path=${base_path}'train_aug/image/'
# val_path=${base_path}'test_ori/image'
# batch_size=32  ##
# visible_device=1  ##
# startpoint=10  ##
# encoder='timm-skresnext50_32x4d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='timm-resnest50d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='resnext50_32x4d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='resnet50' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 


# base_path='./dataset3/4/'
# base_save_path='./results/dataset3/4/model/'

# train_path=${base_path}'train_aug/image/'
# val_path=${base_path}'test_ori/image'
# batch_size=32  ##
# visible_device=1  ##
# startpoint=10  ##
# encoder='timm-skresnext50_32x4d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='timm-resnest50d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='resnext50_32x4d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='resnet50' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 


# base_path='./dataset3/5/'
# base_save_path='./results/dataset3/5/model/'

# train_path=${base_path}'train_aug/image/'
# val_path=${base_path}'test_ori/image'
# batch_size=32  ##
# visible_device=1  ##
# startpoint=10  ##
# encoder='timm-skresnext50_32x4d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='timm-resnest50d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='resnext50_32x4d' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

# encoder='resnet50' #timm-skresnext50_32x4d
# attention=None
# save_path=${base_save_path}'cotrain_dice_traug_teori_0210_w311_cat_2fea_'${encoder}
# classnum=3
# pretrain='imagenet'

# NVIDIA_ENABLE_TF32=1 NVIDIA_TF32_OVERRIDE=1 CUDA_VISIBLE_DEVICES=${visible_device} python train_seg_clf.py \
# --train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
# --object_type ${object_type} --save_path ${save_path} --loss_type ${loss_type} \
# --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
# --batch_size ${batch_size} --LR_seg ${LR_seg} --LR_clf ${LR_clf} --num_epochs ${num_epochs} --startpoint ${startpoint} --classnum ${classnum} 

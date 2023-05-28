usenorm=True
visible_device=0,1,2,3
classnum=3
loss_type='dice'
pretrain='imagenet'
in_channels=5
encoder_depth=4
activation='sigmoid'
test_data_path='/raid5/hehq/MICCAI2021/Test_data/data/'                # done
test_mask_path='/raid5/hehq/MICCAI2021/Test_data/mask/'

# ====================================================================================



# train_data_path='/raid5/hehq/MICCAI2021/Train_data/Train/data/'        # done
train_data_path='/raid5/hehq/MICCAI2021/Train_data/Train/data/'
# train_mask_path='/raid5/hehq/MICCAI2021/Train_data/Train/mask/Disc/'
train_mask_path='/raid5/hehq/MICCAI2021/Train_data/Train/mask/'
val_data_path='/raid5/hehq/MICCAI2021/Train_data/Train/data/'
val_mask_path='/raid5/hehq/MICCAI2021/Train_data/Train/mask/'
model_file='/raid5/hehq/MICCAI2021/task2/code/best_dice_0.9855jaccard_0.9713.pt'    # none
save_path='/raid5/hehq/MICCAI2021/ISBI/model/'

model_type='unet_smp'
encoder='timm-resnest50d'
LR_seg=4e-4
batch_size=16
cuda_no='0'
target='coarse'+${encoder}+${model_type}


# flod=1
# CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
# --test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
# --train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
# --val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
# --model_type ${model_type} --save_path ${save_path} \
# --encoder ${encoder} --pretrain ${pretrain} \
# --classnum ${classnum} --usenorm ${usenorm} \
# --model_file ${model_file} --encoder_depth ${encoder_depth} \
# --activation ${activation} --in_channels ${in_channels} \
# --cuda_no ${cuda_no} --flod ${flod} \
# --LR_seg ${LR_seg} --batch_size ${batch_size} \
# --target ${target} --loss_type ${loss_type}\

# flod=2
# CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
# --test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
# --train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
# --val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
# --model_type ${model_type} --save_path ${save_path} \
# --encoder ${encoder} --pretrain ${pretrain} \
# --classnum ${classnum} --usenorm ${usenorm} \
# --model_file ${model_file} --encoder_depth ${encoder_depth} \
# --activation ${activation} --in_channels ${in_channels} \
# --cuda_no ${cuda_no} --flod ${flod} \
# --LR_seg ${LR_seg} --batch_size ${batch_size} \
# --target ${target} --loss_type ${loss_type}\

# flod=3
# CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
# --test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
# --train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
# --val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
# --model_type ${model_type} --save_path ${save_path} \
# --encoder ${encoder} --pretrain ${pretrain} \
# --classnum ${classnum} --usenorm ${usenorm} \
# --model_file ${model_file} --encoder_depth ${encoder_depth} \
# --activation ${activation} --in_channels ${in_channels} \
# --cuda_no ${cuda_no} --flod ${flod} \
# --LR_seg ${LR_seg} --batch_size ${batch_size} \
# --target ${target} --loss_type ${loss_type}\

# flod=4
# CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
# --test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
# --train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
# --val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
# --model_type ${model_type} --save_path ${save_path} \
# --encoder ${encoder} --pretrain ${pretrain} \
# --classnum ${classnum} --usenorm ${usenorm} \
# --model_file ${model_file} --encoder_depth ${encoder_depth} \
# --activation ${activation} --in_channels ${in_channels} \
# --cuda_no ${cuda_no} --flod ${flod} \
# --LR_seg ${LR_seg} --batch_size ${batch_size} \
# --target ${target} --loss_type ${loss_type}\

# flod=5
# CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
# --test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
# --train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
# --val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
# --model_type ${model_type} --save_path ${save_path} \
# --encoder ${encoder} --pretrain ${pretrain} \
# --classnum ${classnum} --usenorm ${usenorm} \
# --model_file ${model_file} --encoder_depth ${encoder_depth} \
# --activation ${activation} --in_channels ${in_channels} \
# --cuda_no ${cuda_no} --flod ${flod} \
# --LR_seg ${LR_seg} --batch_size ${batch_size} \
# --target ${target} --loss_type ${loss_type}\


model_type='unet_smp'
encoder='timm-efficientnet-b4'
LR_seg=4e-4
batch_size=16
cuda_no='0'
target='coarse'+${encoder}+${model_type}


# flod=1
# CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
# --test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
# --train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
# --val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
# --model_type ${model_type} --save_path ${save_path} \
# --encoder ${encoder} --pretrain ${pretrain} \
# --classnum ${classnum} --usenorm ${usenorm} \
# --model_file ${model_file} --encoder_depth ${encoder_depth} \
# --activation ${activation} --in_channels ${in_channels} \
# --cuda_no ${cuda_no} --flod ${flod} \
# --LR_seg ${LR_seg} --batch_size ${batch_size} \
# --target ${target} --loss_type ${loss_type}\

# flod=2
# CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
# --test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
# --train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
# --val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
# --model_type ${model_type} --save_path ${save_path} \
# --encoder ${encoder} --pretrain ${pretrain} \
# --classnum ${classnum} --usenorm ${usenorm} \
# --model_file ${model_file} --encoder_depth ${encoder_depth} \
# --activation ${activation} --in_channels ${in_channels} \
# --cuda_no ${cuda_no} --flod ${flod} \
# --LR_seg ${LR_seg} --batch_size ${batch_size} \
# --target ${target} --loss_type ${loss_type}\

# flod=3
# CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
# --test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
# --train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
# --val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
# --model_type ${model_type} --save_path ${save_path} \
# --encoder ${encoder} --pretrain ${pretrain} \
# --classnum ${classnum} --usenorm ${usenorm} \
# --model_file ${model_file} --encoder_depth ${encoder_depth} \
# --activation ${activation} --in_channels ${in_channels} \
# --cuda_no ${cuda_no} --flod ${flod} \
# --LR_seg ${LR_seg} --batch_size ${batch_size} \
# --target ${target} --loss_type ${loss_type}\

flod=1
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=5
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\



model_type='unet++'
encoder='timm-resnest50d'
LR_seg=4e-4
batch_size=16
cuda_no='0'
target='coarse'+${encoder}+${model_type}


flod=1
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=2
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=3
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=4
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=5
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\


model_type='unet++'
encoder='timm-efficientnet-b4'
LR_seg=4e-4
batch_size=16
cuda_no='0'
target='coarse'+${encoder}+${model_type}


flod=1
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=2
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=3
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=4
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\

flod=5
CUDA_VISIBLE_DEVICES=${visible_device} python faz_segmentation.py \
--test_data_path ${test_data_path} --test_mask_path ${test_mask_path} \
--train_data_path ${train_data_path} --train_mask_path ${train_mask_path} \
--val_data_path ${val_data_path} --val_mask_path ${val_mask_path} \
--model_type ${model_type} --save_path ${save_path} \
--encoder ${encoder} --pretrain ${pretrain} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file} --encoder_depth ${encoder_depth} \
--activation ${activation} --in_channels ${in_channels} \
--cuda_no ${cuda_no} --flod ${flod} \
--LR_seg ${LR_seg} --batch_size ${batch_size} \
--target ${target} --loss_type ${loss_type}\
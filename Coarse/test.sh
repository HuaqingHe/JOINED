loss_type='dice'
pretrain='imagenet'
dist_type='dist_mask'
train_path='/home/haoyum/download/Coarse/data/train'
val_path='/home/haoyum/download/Coarse/data/val'
save_path='/home/haoyum/download/Coarse/result'
visible_device=0,1,2,3,4,5,6,7

# =======================================================================================

batch_size=16
LR_seg=1e-4
num_epoch=100
aux=False
model_type='unet++doublesmp'
encoder='timm-resnet50d'
model_file='/raid5/hehq/MICCAI2021/REFUGE/Coarse/_newsave_9fovea=0.05hDist=1/best_ODdice_0.8093best_OCdice_0.6519_DCjaccard_0.5845_FMSE_204.8512_DMSE_150.4663_FDdistloss_0.0053_epoch_150.pt'

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_seg_smp_MSE.py --loss_type $loss_type --pretrain $pretrain \
--dist_type $dist_type --batch_size $batch_size --LR_seg $LR_seg \
--num_epoch $num_epoch --aux $aux --model_type $model_type --encoder $encoder \
--train_path $train_path --val_path $val_path --save_path $save_path

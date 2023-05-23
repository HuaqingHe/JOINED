pretrain='imagenet'
distance_type='dist_signed11'
usenorm=True
visible_device=2
classnum=3
loss='dice'


model_type='unet_smp'
encoder='timm-resnest50d'

base_path='./dataset3/5/'
train_type='aug/'
test_type='ori/'

file_output='./results/paper_results/seg_best/dataset3'
train_path=${base_path}'train_'${train_type}'image/'
val_path=${base_path}'test_'${test_type}'image'
model_file='
./results_seg/dataset3/5/aug/unet_smp_imagenet_dice_timm-resnest50d_new/best_dice_0.9645_jaccard_0.9315.pt
'

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_seg.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file}

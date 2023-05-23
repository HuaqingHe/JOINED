model_type='cotrain'

encoder='timm-resnest50d'
pretrain='imagenet'
distance_type='dist_signed11'

usenorm=True

base_path='./dataset3/5/'
# base_save_path='./results/dataset2/5/model/'

train_path=${base_path}'train_aug/image/'
val_path=${base_path}'test_ori/image'
visible_device=2  ##
model_file='./results/dataset3/5//model/cotrain_dice_traug_teori_0210_w311_cat_2fea_timm-resnest50d/dice_0.96058_jaccard_0.92414_acc_0.9028_kap_0.7895.pt
'

classnum=3
file_output='./results/paper_results/bsda_seg_wob/dataset3'

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} \
--model_file ${model_file}

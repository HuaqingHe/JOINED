object_type='faz'
loss='dice'
distance_type='dist_mask'
LR_seg=1e-4
num_epochs=100
visible_device=0
usenorm=True
aux='False'
encode='vgg16'
classnum=3
dataset=3

pretrain='True'
train_type='ori/'
test_type='ori/'

# ====================================================================================

model_type='vgg16'
encoder='vgg16'

file_output='./output_results/dataset'${dataset}'/clf/'${train_type}${model_type}'_'${pretrain}'/'

base_path='../dataset'${dataset}'/XXX/'
base_save_path='./results'${dataset}'/XXX/cls/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'image/'
val_path=${base_path}'test_'${test_type}'image'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file} --classnum ${classnum}

model_type='resnet50'
encoder='vgg16'

file_output='./output_results/dataset'${dataset}'/clf/'${train_type}${model_type}'_'${pretrain}'/'

base_path='../dataset'${dataset}'/XXX/'
base_save_path='./results'${dataset}'/XXX/cls/'${train_type}${model_type}'_'${pretrain}'_'${loss}'_'${encoder}'/'
train_path=${base_path}'train_'${train_type}'image/'
val_path=${base_path}'test_'${test_type}'image'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file} --classnum ${classnum}

model_type='resnest50'
encoder='vgg16'

file_output='./output_results/dataset'${dataset}'/clf/'${train_type}${model_type}'_'${pretrain}'/'

base_path='../dataset'${dataset}'/XXX/'
base_save_path='./results'${dataset}'/XXX/cls/'${train_type}${model_type}'_'${pretrain}'/'
train_path=${base_path}'train_'${train_type}'image/'
val_path=${base_path}'test_'${test_type}'image'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file} --classnum ${classnum}

model_type='resnext50'
encoder='vgg16'

file_output='./output_results/dataset'${dataset}'/clf/'${train_type}${model_type}'_'${pretrain}'/'

base_path='../dataset'${dataset}'/XXX/'
base_save_path='./results'${dataset}'/XXX/cls/'${train_type}${model_type}'_'${pretrain}'/'
train_path=${base_path}'train_'${train_type}'image/'
val_path=${base_path}'test_'${test_type}'image'
model_file=${base_save_path}

CUDA_VISIBLE_DEVICES=${visible_device} python evaluate_clf.py \
--train_path ${train_path} --val_path ${val_path} --model_type ${model_type} \
--save_path ${file_output} --encoder ${encoder} --pretrain ${pretrain} --distance_type ${distance_type} \
--classnum ${classnum} --usenorm ${usenorm} --model_file ${model_file} --classnum ${classnum}

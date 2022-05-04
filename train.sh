
device=3,2,1
epochs=10
batch_size=128
raw_data_dir=../contest_data/testA/train
pretrain_dir=../pretrain_model
length=-1

# bert_path=$pretrain_dir/model/bert-base-chinese
# vbert_path=$pretrain_dir/model/visualbert-vqa-coco-pre
# tokenizer_path=$pretrain_dir/tokenizer/bert-base-chinese

bert_path=bert-base-chinese
vbert_path=uclanlp/visualbert-nlvr2-coco-pre
tokenizer_path=bert-base-chinese

# train setting
config_path='./config.json'
# save_dir='../best_model/'
save_dir='../data/model_data/8layer_drop_attr_match0_jieba'

# train model
cd code && python3 train.py \
    -c=$config_path \
    -d=$device \
    --epochs=$epochs \
    --batch_size=$batch_size \
    --save_dir=$save_dir \
    --bert_path=$bert_path \
    --vbert_path=$vbert_path \
    --tokenizer_path=$tokenizer_path \
    --raw_data_dir=$raw_data_dir \
    --data_length=$length

# move back
cd ..

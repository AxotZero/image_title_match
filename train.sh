
device=4
epochs=1
batch_size=32
raw_data_dir=../contest_data/testA/train
pretrain_dir=../pretrain_model
length=1000

bert_path=$pretrain_dir/model/bert-base-chinese
vbert_path=$pretrain_dir/model/visualbert-nlvr2-coco-pre
tokenizer_path=$pretrain_dir/tokenizer/bert-base-chinese

# train setting
config_path='./config.json'
save_dir='../best_model/'

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
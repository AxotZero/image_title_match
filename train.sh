
device=5,4
epochs=10
batch_size=64
raw_data_dir=../contest_data/testA/train
pretrain_dir=../pretrain_model
length=-1

bert_path=$pretrain_dir/model/bert-base-chinese
vbert_path=$pretrain_dir/model/visualbert-nlvr2-coco-pre
tokenizer_path=$pretrain_dir/tokenizer/bert-base-chinese

# bert_path=bert-base-chinese
# vbert_path=uclanlp/visualbert-nlvr2-coco-pre
# tokenizer_path=bert-base-chinese

# train setting
config_path='./config.json'
# save_dir='../best_model/'
# resume_path='../data/model_data/img_enhance_vbert_layer8_loop3/best_model.pth'
save_dir='../data/model_data/img_enhance_vbert_layer8_loop2_all_data'

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
    --data_length=$length \
    # --resume=$resume_path

# move back
cd ..

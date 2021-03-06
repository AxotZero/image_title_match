
device=2,1
epochs=12
batch_size=64
raw_data_dir=../contest_data/testA/train
pretrain_dir=../pretrain_model
length=1000

# bert_path=$pretrain_dir/model/bert-base-chinese
# vbert_path=$pretrain_dir/model/visualbert-nlvr2-coco-pre
# tokenizer_path=$pretrain_dir/tokenizer/bert-base-chinese

bert_path=bert-base-chinese
vbert_path=uclanlp/visualbert-nlvr2-coco-pre
tokenizer_path=bert-base-chinese

# train setting
# config_path='./config.json'
# save_dir='../best_model/'
# resume_path='../data/model_data/img_enhance_vbert_layer8_loop3/best_model.pth'
# save_dir='../data/model_data/visual_mask_all_layer7'


for config_path in  'config/no_visual_proj_nsplit12_drop_no_attr.json';
do  
    echo '=== run' $config_path '==='
    # train model
    cd code && python3 train.py \
        -c=$config_path \
        -d=$device \
        --bs=$batch_size \
        --data_length=$length \
        --epochs=$epochs \
        --raw_data_dir=$raw_data_dir \
        --data_length=$length \
        # --batch_size=$batch_size \
        # --save_dir=$save_dir \
        # --bert_path=$bert_path \
        # --vbert_path=$vbert_path \
        # --tokenizer_path=$tokenizer_path \
        # --resume=$resume_path
    # move back
    cd ..
done

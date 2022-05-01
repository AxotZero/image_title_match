# move to working directory


# test setting
device=4
save_dir='../data/tmp_data/'
raw_data_dir=../contest_data
attr_path=$raw_data_dir/train/attr_to_attrvals.json
test_data_path=$raw_data_dir/semi_testA.txt

pretrain_dir=../pretrain_model

text_model_name=$pretrain_dir/model/bert-base-chinese

# run test
cd code && python3 test.py \
    --device=$device \
    --resume=../data/best_model.pth \
    --output_dir=$save_dir \
    --text_model_name=$text_model_name \
    --attr_path=$attr_path \
    --test_data_path=$test_data_path

# make submission
cd code && python3 ./merge_probs.py \
    --save_dirs $save_dir

# move back
cd ..
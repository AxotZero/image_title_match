# 代码说明

## 环境配置
注明python、pytorch等依赖的版本，并对init.sh每一步进行描述，或者在init.sh中对每一步添加注释
- 作业系统: ubuntu 18.04
- python: 3.7.13
- pytorch: 1.9.0
- cuda: 10.2

### init.sh 解释

```sh=
# install python3.7
apt update
apt -y install software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt -y install python3.7

# change the default python3 to python3.7
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# install pip
apt-get update
apt-get -y install python3-pip
pip3 install --upgrade pip


# install git
apt install git -y


# install requirements
yes | pip3 install -r requirements.txt
```

## 数据
仅使用主办单位的数据

## 预训练模型
使用两个 huggingface 上的预训练模型，分别为:
- bert-base-chinese
- uclanlp/visualbert-vqa-coco-pre

## 算法

### 整体思路介绍
1. 模型输入输出
    - 将关键属性与文本标题都当作模型文字输入来给模型分类是否匹配
3. 模型训练
    - coarse 里虽然没有关键属性标签但依然有图文是否匹配的资讯，可以先选择先从标题里面抽取出属性，并将其属性的标签设定为图文是否匹配
        - 若图文为 0 ，则其属性标签则全为 0，反之亦然
    - 训练时，一定机率改变文本中的属性，并将匹配改为 0
4. 测试
    - 将关键属性与文本标题都当作模型文字输入来给模型分类，并以阀值 0.5 来决定是否匹配

### 网络结构
1. 将文字输入到 bert-base-chinese 中获得所有文字的 text_embedding
2. 将前一步骤的 text_embedding 与主办提供的图片 feature 一起输入到 visual_bert 得到 visual_bert pool 后的输出
3. 将 pool 后的输出拿去分类是否匹配

### 损失函数
torch.nn.BCELoss()

### 数据扩增
- 文本标题
    - 图文匹配为 1 时，以 0.5 的机率做以下操作:
        1. 图文匹配改为 0
        2. 随机选取一个关键属性替换
        3. 其余关键属性以一定机率替换或删除
    - 图文匹配为 0 : 不做任何扩增
- 关键属性
    - 匹配为 1 时，以0.5机率替换为其他属性 

### 模型集成
无。


## 训练流程
### 使用范例
```sh=
. ./train.sh \
    0 \
    ./data/contest_data/train
```
- $1: gpu_id
- $2: 官方数据中存放 `train_coarse.txt`, `train_fine.txt`, `attr_to_attrvals.json` 原始资料集的资料夹

### train.sh 解释

```sh=
# move to working directory
cd code

# preprocess data
echo "=== preprocess train data ==="
raw_data_dir=../$2
train_data_dir=../data/tmp_data
python3 process_data.py \
    --coarse_data_path=$raw_data_dir/train_coarse.txt \
    --fine_data_path=$raw_data_dir/train_fine.txt \
    --attr_to_attrvals_path=$raw_data_dir/attr_to_attrvals.json \
    --save_dir=$train_data_dir


# train setting
echo "=== train model ==="
device=$1
epochs=1
text_model_name='bert-base-chinese'
save_dir='../data/model_data/model1/'
config_path='./config.json'

# train model
python3 train.py \
    -c=$config_path \
    -d=$device \
    --ep=$epochs \
    --save_dir=$save_dir \
    --dt=$text_model_name \
    --mt=$text_model_name \
    --dp=$train_data_dir

# move back
cd ..
```

## 测试流程（必选）
### 使用范例
```sh=
. ./test.sh \
    0 \
    ./data/contest_data/train/attr_to_attrvals.json \
    ./data/contest_data/preliminary_testB.txt
```
- $1: gpu_id
- $2: 官方数据中原始 `attr_to_attrvals.json` 的路径
- $3: 官方测试数据的路径

### test.sh 解释
```sh=
# move to working directory
cd code

# test setting
device=$1
text_model_name='bert-base-chinese'
save_dir='../data/tmp_data/'

# run test
echo "=== run test ==="

python3 test.py \
    --device=$device \
    --resume=../data/best_model.pth \
    --output_dir=$save_dir \
    --text_model_name=$text_model_name \
    --attr_path=../$2 \
    --test_data_path=../$3


# make submission
python3 ./merge_probs.py \
    --save_dirs $save_dir

# move back
cd ..
```


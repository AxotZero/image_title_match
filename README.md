# 电商关键属性的图文匹配競賽
* [比賽連結](https://www.heywhale.com/home/competition/620b34c41f3cf500170bd6ca)
* Public LB: 18/1467
* Private LB: 30/1467

## Competition input, output
- input
    - img_feature: 2048 維的特徵
    - 商品標題
- output
    - 各個關鍵屬性和整體圖文是否匹配 (0/1)
- 備註
    - 一個標題會有 0~N 個關鍵屬性
    - 一定會有圖文是否匹配的標記

## Pretrained model
使用两个 huggingface 上的预训练模型，分别为:
- hfl/chinese-macbert-large
- uclanlp/visualbert-nlvr2-coco-pre

## Model Architecture
- 拿 macbert 的 word_embedding 及 encoder layers 直接 assign 到 visual_bert 的 word_embedding 和 encoder 中
    - <img src="https://i.imgur.com/n5Xlfc3.png" width=550 height=130>
-  拿 img_feature 用一層 NN 來產生 12 種關鍵屬性 + 1 種整體圖文 的 features 來當作 visual_bert 的 image_embeds 輸入
- 模型輸入輸出的方法
    - <img src="https://i.imgur.com/yZeyjxM.png" width=600 height=150>


### 方法
- 將關鍵屬性與文本標題都當作模型文字輸入來給模型分類是否匹配
- 丟關鍵屬性時
    - 產生一個把其他 image_embeds 都 mask，只保留該關鍵屬性的 attention_mask 丟給模型
- 丟整個文本標題時
    - 只保留該標題所有關鍵屬性+整體圖文的 attention_mask 丟給模型
- 模型做二分類


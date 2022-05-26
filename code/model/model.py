from pdb import set_trace as bp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from base import BaseModel
from process_data import load_json, load_attr


class VisualBertFromBertBaseChinese2(BaseModel):
    def __init__(self, 
                 bert_path='', vbert_path='', 
                 n_split=13, num_layer=12, num_loop=1, 
                 img_dropout=0.2, attr_img_dropout=0,
                 visual_emb_dim=1024, visual_proj=True, 
                 **kwargs):
        super().__init__()
        self.n_split = n_split
        bert = AutoModel.from_pretrained(bert_path)

        
        self.visual_bert = AutoModel.from_pretrained(vbert_path)
        self.visual_bert.embeddings.word_embeddings = bert.embeddings.word_embeddings
        self.visual_bert.encoder.layer = nn.ModuleList([
            bert.encoder.layer[i%num_layer]
            for i in range(num_layer*num_loop)
        ])
        self.visual_bert.config.num_hidden_layers = num_layer*num_loop
        if not visual_proj:
            visual_emb_dim = 768
            self.visual_bert.embeddings.visual_projection = nn.Identity()

        self.img_encoder = nn.Sequential(
            nn.Dropout(img_dropout),
            nn.utils.weight_norm(nn.Linear(2048, n_split*visual_emb_dim)),
            nn.Dropout(attr_img_dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x) == 2:
            text_ids, img_feature = x
            text_mask = None
            visual_mask = None

        elif len(x) == 3:
            text_ids, img_feature, visual_mask = x
            text_mask = None
        
        elif len(x) == 4:
            text_ids, img_feature, text_mask, visual_mask = x

        if visual_mask:
            visual_mask = visual_mask[:, :self.n_split]

        img_feature = self.img_encoder(img_feature).view(-1, self.n_split, 1024)

        embs = self.visual_bert(
            input_ids=text_ids,
            attention_mask=text_mask,
            visual_embeds=img_feature,
            visual_attention_mask=visual_mask
        )[0][:, -13:]

        return self.classifier(embs).squeeze()



class VisualBertFromBertBaseChinese(BaseModel):
    def __init__(self, 
                 bert_path='', vbert_path='', 
                 n_split=2, num_layer=12, num_loop=1, 
                 img_dropout=0.2, attr_img_dropout=0, 
                 visual_emb_dim=1024, visual_proj=True, 
                 freeze_bert=0,
                 **kwargs):
        super().__init__()
        self.n_split = n_split
        self.visual_emb_dim = visual_emb_dim

        bert = AutoModel.from_pretrained(bert_path)
        self.visual_bert = AutoModel.from_pretrained(vbert_path)
        self.visual_bert.embeddings.word_embeddings = bert.embeddings.word_embeddings
        self.visual_bert.encoder.layer = nn.ModuleList([
            bert.encoder.layer[i%num_layer]
            for i in range(num_layer*num_loop)
        ])
        self.visual_bert.config.num_hidden_layers = num_layer*num_loop
        if not visual_proj:
            self.visual_emb_dim = 768
            self.visual_bert.embeddings.visual_projection = nn.Identity()
        
        # if freeze_bert: 
        for param in self.visual_bert.parameters():
            param.requires_grad = not bool(freeze_bert)


        self.img_encoder = nn.Sequential(
            nn.Dropout(img_dropout),
            nn.utils.weight_norm(nn.Linear(2048, n_split*self.visual_emb_dim)),
            nn.Dropout(attr_img_dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x) == 2:
            text_ids, img_feature = x
            text_mask = None
            visual_mask = None

        elif len(x) == 3:
            text_ids, img_feature, visual_mask = x
            text_mask = None
        
        elif len(x) == 4:
            text_ids, img_feature, text_mask, visual_mask = x

        visual_mask = visual_mask[:, :self.n_split]


        img_feature = self.img_encoder(img_feature).view(-1, self.n_split, self.visual_emb_dim)

        emb = self.visual_bert(
            input_ids=text_ids,
            attention_mask=text_mask,
            visual_embeds=img_feature,
            visual_attention_mask=visual_mask
        )[1]
        return self.classifier(emb).squeeze()


class ImageEnhanceVisualBertFromBertBaseChinese(BaseModel):
    def __init__(self, 
                 bert_path='', vbert_path='', attr_path='', 
                 num_layer=8, num_loop=1, 
                 img_dropout=0.2, img_num_layer=1):
        super().__init__()
        attr_config = load_attr(attr_path)

        bert = AutoModel.from_pretrained(bert_path)
        self.visual_bert = AutoModel.from_pretrained(vbert_path)
        self.visual_bert.embeddings.word_embeddings = bert.embeddings.word_embeddings
        self.visual_bert.encoder.layer = nn.ModuleList([
            bert.encoder.layer[i%num_layer]
            for i in range(num_layer*num_loop)
        ])
        self.visual_bert.config.num_hidden_layers = num_layer*num_loop

        self.img_encoder = nn.Sequential(
            nn.Dropout(img_dropout),
            nn.utils.weight_norm(nn.Linear(2048, 13*1024))
        )

        self.img_classifiers = nn.ModuleList([
            nn.Linear(1024, num_classes)
            for num_classes in attr_config['attr_num_classes'].values()
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # parse data
        if len(x) == 2:
            text_ids, img_feature = x
            text_mask = None
            visual_mask = None

        elif len(x) == 3:
            text_ids, img_feature, visual_mask = x
            text_mask = None
        
        # else:
        #     text_ids, text_mask, img_feature, visual_mask = x


        # encode img
        img_feature = self.img_encoder(img_feature).view(-1, 13, 1024)

        # classify img class
        attr_outputs = [m(img_feature[:, i]) for i, m in enumerate(self.img_classifiers)]


        # classify is match
        emb = self.visual_bert(
            input_ids=text_ids,
            attention_mask=text_mask,
            visual_embeds=img_feature,
            visual_attention_mask=visual_mask
        )[1]
        return self.classifier(emb).squeeze(), attr_outputs


    def predict(self, x):
        return self.forward(x)[0]



class ImageEnhanceVBert(BaseModel):
    def __init__(self, bert_path='', vbert_path='', attr_path=''):
        super().__init__()

        attr_config = load_attr(attr_path)

        self.text_encoder = AutoModel.from_pretrained(bert_path)
        self.visual_bert = AutoModel.from_pretrained(vbert_path)

        self.img_encoder = nn.Linear(2048, 13*1024)

        self.img_classifiers = nn.ModuleList([
            nn.Linear(1024, num_classes)
            for num_classes in attr_config['attr_num_classes'].values()
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # parse data
        if len(x) == 2:
            text_ids, img_feature = x
            text_mask = None
        else:
            text_ids, text_mask, img_feature = x

        # encode text
        text_emb = self.text_encoder(text_ids, text_mask)[0]
        
        # encode img
        img_feature = self.img_encoder(img_feature).view(-1, 13, 1024)

        # classify img class
        attr_outputs = [m(img_feature[:, i]) for i, m in enumerate(self.img_classifiers)]


        # classify is match
        emb = self.visual_bert(
            inputs_embeds=text_emb,
            attention_mask=text_mask,
            visual_embeds=img_feature
        )[1]
        return self.classifier(emb).squeeze(), attr_outputs


class MyVisualBertNlvr2(BaseModel):
    def __init__(self, bert_path='', vbert_path='', n_split=2, bert_num_layer=12, vbert_num_layer=12, **kwargs):
        super().__init__()
        self.n_split = n_split
        self.text_encoder = AutoModel.from_pretrained(bert_path)
        self.text_encoder.encoder.layer = self.text_encoder.encoder.layer[:bert_num_layer]


        self.visual_bert = AutoModel.from_pretrained(vbert_path)
        self.visual_bert.encoder.layer = self.visual_bert.encoder.layer[:vbert_num_layer]

        self.img_encoder = nn.Linear(2048, n_split*1024)

        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x) == 2:
            text_ids, img_feature = x
            text_mask = None
        else:
            text_ids, text_mask, img_feature = x

        text_emb = self.text_encoder(text_ids, text_mask)[0]
        img_feature = self.img_encoder(img_feature)

        emb = self.visual_bert(
            inputs_embeds=text_emb,
            attention_mask=text_mask,
            # visual_embeds=img_feature.unsqueeze(1)
            visual_embeds=img_feature.view(-1, self.n_split, 1024)
        )[1]
        return self.classifier(emb).squeeze()


class MyVisualBert(BaseModel):
    def __init__(self, bert_path='', vbert_path=''):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(bert_path)
        self.visual_bert = AutoModel.from_pretrained(vbert_path)

        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x) == 2:
            text_ids, img_feature = x
            text_mask = None
        else:
            text_ids, text_mask, img_feature = x

        text_emb = self.text_encoder(text_ids, text_mask)[0]

        emb = self.visual_bert(
            inputs_embeds=text_emb,
            attention_mask=text_mask,
            visual_embeds=img_feature.unsqueeze(1)
        )[1]
        return self.classifier(emb).squeeze()


class QueryModel(BaseModel):
    def __init__(self, text_model_name='bert-base-chinese'):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_lower_dim = nn.Linear(768, 128)

        self.img_encoder = nn.Sequential(
            nn.Dropout(0.2),

            nn.Linear(2048, 1536),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(1536, 128 * 13)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),

            nn.Linear(2048 + 768, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        text_ids, img_feature = x
        text_emb = self.text_encoder(text_ids)[0][:, 0]  # cls
        emb = torch.cat((text_emb, img_feature), dim=1)
        return self.classifier(emb).squeeze()


class FirstPlaceModelFreeze(BaseModel):
    def __init__(self, text_model_name='bert-base-chinese'):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(768, 4, 2048, 0.2, batch_first=True),
            num_layers=3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),

            nn.Linear(2048 + 768, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        text_ids, img_feature = x
        text_emb = self.text_encoder(text_ids)[0]
        text_emb = self.attn(text_emb)[:, 0]
        emb = torch.cat((text_emb, img_feature), dim=1)
        return self.classifier(emb).squeeze()


class FirstPlaceModel(BaseModel):
    def __init__(self, text_model_name='bert-base-chinese'):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),

            nn.Linear(2048 + 768, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x) == 2:
            text_ids, img_feature = x
            text_mask = None
        else:
            text_ids, text_mask, img_feature = x
        
        text_emb = self.text_encoder(text_ids, text_mask)[0][:, 0]  # cls
        emb = torch.cat((text_emb, img_feature), dim=1)
        return self.classifier(emb).squeeze()


class ImageClassifierD4(BaseModel):
    def __init__(self, preprocess_dir, text_model_name='bert-base-chinese'):
        super().__init__()
        attr_config = load_json(f'{preprocess_dir}/attr_config.json')

        # embs_logits
        self.text_model = AutoModel.from_pretrained(text_model_name)

        self.img_encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 768),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(2048, 512),
                nn.LeakyReLU(),
                nn.Linear(512, d['num_classes'])
            )
            for d in attr_config['attr_num_classes'].values()]
        )
        self.update_threshold(0.5)

    def update_threshold(self, t):
        self.threshold = nn.Parameter(torch.tensor(t))

    def forward(self, x):
        title_ids, img_feature = x

        # get emb
        title_emb = self.text_model(title_ids)[0][:, 0]
        img_emb = self.img_encoder(img_feature)

        # normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        title_emb = title_emb / title_emb.norm(dim=-1, keepdim=True)

        # cosine similarity
        logit_scale = self.logit_scale.exp()
        embs_logit = logit_scale * img_emb @ title_emb.t()
        # logits_per_text = logits_per_image.t()

        attr_outputs = [m(img_feature) for m in self.classifiers]
        return embs_logit, attr_outputs


class ImageClassifierD3(BaseModel):
    def __init__(self, preprocess_dir, text_model_name='bert-base-chinese'):
        super().__init__()
        attr_config = load_json(f'{preprocess_dir}/attr_config.json')

        # embs_logits
        self.text_model = AutoModel.from_pretrained(text_model_name)
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.img_encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 768),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(2048, 512),
                nn.LeakyReLU(),
                nn.Linear(512, d['num_classes'])
            )
            for d in attr_config['attr_num_classes'].values()]
        )
        self.update_threshold(0.5)

    def update_threshold(self, t):
        self.threshold = nn.Parameter(torch.tensor(t))

    def forward(self, x):
        title_ids, img_feature = x

        # get emb
        title_emb = self.text_model(title_ids)[0][:, 0]
        img_emb = self.img_encoder(img_feature)

        # normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        title_emb = title_emb / title_emb.norm(dim=-1, keepdim=True)

        # cosine similarity
        logit_scale = self.logit_scale.exp()
        embs_logit = logit_scale * img_emb @ title_emb.t()
        # logits_per_text = logits_per_image.t()

        attr_outputs = [m(img_feature) for m in self.classifiers]
        return embs_logit, attr_outputs


class ImageClassifierD2(BaseModel):
    def __init__(self, preprocess_dir, text_model_name='bert-base-chinese'):
        super().__init__()
        attr_config = load_json(f'{preprocess_dir}/attr_config.json')

        # embs_logits
        self.text_model = AutoModel.from_pretrained(text_model_name)
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.text_encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(768, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 512, 0.2, batch_first=True),
            num_layers=2
        )
        self.img_encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(2048, 512),
                nn.LeakyReLU(),
                nn.Linear(512, d['num_classes'])
            )
            for d in attr_config['attr_num_classes'].values()]
        )
        self.update_threshold(0.5)

    def update_threshold(self, t):
        self.threshold = nn.Parameter(torch.tensor(t))

    def forward(self, x):
        title_ids, img_feature = x

        # get emb
        title_embs = self.text_model(title_ids)[0]
        title_embs = self.text_encoder(title_embs)
        title_emb = self.attn(title_embs)[:, 0]
        img_emb = self.img_encoder(img_feature)

        # normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        title_emb = title_emb / title_emb.norm(dim=-1, keepdim=True)

        # cosine similarity
        logit_scale = self.logit_scale.exp()
        embs_logit = logit_scale * img_emb @ title_emb.t()
        # logits_per_text = logits_per_image.t()

        attr_outputs = [m(img_feature) for m in self.classifiers]
        return embs_logit, attr_outputs


class ImageClassifierD1(BaseModel):
    def __init__(self, preprocess_dir, text_model_name='bert-base-chinese'):
        super().__init__()
        attr_config = load_json(f'{preprocess_dir}/attr_config.json')

        # embs_logits
        self.text_model = AutoModel.from_pretrained(text_model_name)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(768, 4, 2048, 0.2, batch_first=True),
            num_layers=2
        )

        self.text_encoder = nn.Linear(768, 512)
        self.img_encoder = nn.Linear(2048, 512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.15),
                nn.Linear(2048, d['num_classes'])
            )
            for d in attr_config['attr_num_classes'].values()]
        )
        self.update_threshold(0.5)

    def update_threshold(self, t):
        self.threshold = nn.Parameter(torch.tensor(t))

    def forward(self, x):
        title_ids, img_feature = x

        # get emb
        title_embs = self.text_model(title_ids)[0]
        title_emb = self.attn(title_embs)[:, 0]
        title_emb = self.text_encoder(title_emb)
        img_emb = self.img_encoder(img_feature)

        # normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        title_emb = title_emb / title_emb.norm(dim=-1, keepdim=True)

        # cosine similarity
        logit_scale = self.logit_scale.exp()
        embs_logit = logit_scale * img_emb @ title_emb.t()
        # logits_per_text = logits_per_image.t()

        attr_outputs = [m(img_feature) for m in self.classifiers]
        return embs_logit, attr_outputs


class ImageClassifier(BaseModel):
    def __init__(self, preprocess_dir, text_model_name='bert-base-chinese'):
        super().__init__()
        attr_config = load_json(f'{preprocess_dir}/attr_config.json')

        self.text_model = AutoModel.from_pretrained(text_model_name)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_encoder = nn.Linear(768, 512)
        self.img_encoder = nn.Linear(2048, 512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.classifiers = nn.ModuleList([
            nn.Linear(2048, d['num_classes'])
            for d in attr_config['attr_num_classes'].values()]
        )
        self.threshold = 0.5

    def update_threshold(self, t):
        self.threshold = t

    def forward(self, x):
        title_ids, img_feature = x

        # get emb
        title_emb = self.text_model(title_ids)[0][:, 0]
        title_emb = self.text_encoder(title_emb)
        img_emb = self.img_encoder(img_feature)

        # normalize
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        title_emb = title_emb / title_emb.norm(dim=-1, keepdim=True)

        # cosine similarity
        logit_scale = self.logit_scale.exp()
        embs_logit = logit_scale * img_emb @ title_emb.t()
        # logits_per_text = logits_per_image.t()

        attr_outputs = [m(img_feature) for m in self.classifiers]
        return embs_logit, attr_outputs


class B2Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.title_encoder = BaseTitleEncoder(
            emb_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3
        )
        self.global_classifier = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.attr_classifier = nn.Sequential(
            nn.Linear(128, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        title_ids, img_features = x
        title_emb = self.title_encoder(title_ids)
        # img_emb = self.img_encoder(img_features)
        # emb = torch.cat((title_emb, img_emb), dim=1)
        global_pred = self.global_classifier(title_emb).view(-1)
        attr_pred = self.attr_classifier(title_emb)
        return (global_pred, attr_pred)


class QueryTitleEncoder(nn.Module):
    def __init__(self, num_vocab=2000, emb_size=768, num_layers=3, dropout=0.3):
        super().__init__()
        self.emb_size = emb_size
        self.embedder = nn.Embedding(num_vocab, emb_size)
        self.query_embedding = nn.Parameter(torch.randn(13, emb_size))
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                emb_size, 4, emb_size * 4, dropout, batch_first=True),
            num_layers
        )
        self.pooling_layer = nn.AdaptiveAvgPool1d(self.emb_size)

    def forward(self, title_ids):
        bs = len(title_ids)
        query_embs = self.query_embedding.repeat(bs, 1, 1)  # bs, 13, emb_size
        title_embs = self.embedder(title_ids)
        title_embs = self.attn(title_embs)  # bs, n_vocab, emb_size
        title_emb = self.pooling_layer(title_embs.view(bs, 1, -1)).view(bs, -1)
        return title_emb


class BaselineSplitModel(BaseModel):
    def __init__(self, title_emb_size=32, img_hidden_size=128, img_emb_size=32):
        super().__init__()

        self.attr_match_models = nn.ModuleList(
            [
                SingleMatchModel(
                    title_emb_size=title_emb_size,
                    img_hidden_size=img_hidden_size,
                    img_emb_size=img_emb_size)
                for _ in range(12)
            ]
        )
        self.global_match_model = SingleMatchModel(
            title_emb_size=128,
            img_hidden_size=512,
            img_emb_size=128,
        )

    def forward(self, x):
        global_pred = self.global_match_model(x).view(-1)
        attr_preds = [model(x) for model in self.attr_match_models]
        attr_preds = torch.cat(attr_preds, dim=1)
        return (global_pred, attr_preds)


class SingleMatchModel(BaseModel):
    def __init__(self, title_emb_size=32, img_hidden_size=128, img_emb_size=32):
        super().__init__()
        self.title_encoder = BaseTitleEncoder(emb_size=title_emb_size)
        self.img_encoder = CnnEncoder(
            hidden_size=img_hidden_size, num_targets=img_emb_size)
        self.classifier = nn.Sequential(
            nn.Linear(title_emb_size+img_emb_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        title_ids, img_features = x
        title_emb = self.title_encoder(title_ids)
        img_emb = self.img_encoder(img_features)
        emb = torch.cat((title_emb, img_emb), dim=1)
        pred = self.classifier(emb)
        return pred


class BModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.title_encoder = BaseTitleEncoder(
            emb_size=128,
            hidden_size=256,
            num_layers=2,
            dropout=0.3
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
        )
        self.global_classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.attr_classifier = nn.Sequential(
            nn.Linear(256, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        title_ids, img_features = x
        title_emb = self.title_encoder(title_ids)
        img_emb = self.img_encoder(img_features)
        emb = torch.cat((title_emb, img_emb), dim=1)
        global_pred = self.global_classifier(emb).view(-1)
        attr_pred = self.attr_classifier(emb)
        return (global_pred, attr_pred)


class BaselineModel(BaseModel):
    def __init__(self, title_emb_size=128, img_emb_size=768):
        super().__init__()
        self.title_emb_size = title_emb_size
        self.title_encoder = BaseTitleEncoder(emb_size=title_emb_size)
        self.img_encoder = CnnEncoder(num_targets=img_emb_size)

        self.global_classifier = nn.Sequential(
            nn.Linear(title_emb_size+img_emb_size, 1),
            nn.Sigmoid()
        )
        self.attr_classifier = nn.Sequential(
            nn.Linear(title_emb_size+img_emb_size, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        title_ids, img_features = x
        title_emb = self.title_encoder(title_ids)
        img_emb = self.img_encoder(img_features)
        emb = torch.cat((title_emb, img_emb), dim=1)
        global_pred = self.global_classifier(emb).view(-1)
        attr_pred = self.attr_classifier(emb)
        return (global_pred, attr_pred)


class BaseTitleEncoder(nn.Module):
    def __init__(self, num_vocab=2000, emb_size=768, hidden_size=768*3, num_layers=3, dropout=0.3):
        super().__init__()
        # 詞的pretrained embedding
        self.emb_size = emb_size
        self.embedder = nn.Embedding(num_vocab, emb_size)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                emb_size, 4, hidden_size, dropout, batch_first=True),
            num_layers
        )
        self.pooling_layer = nn.AdaptiveAvgPool1d(self.emb_size)

    def forward(self, title_ids):
        bs = len(title_ids)
        title_embs = self.embedder(title_ids)
        title_embs = self.attn(title_embs)  # bs, n_vocab, emb_size
        title_emb = self.pooling_layer(title_embs.view(bs, 1, -1)).view(bs, -1)
        return title_emb


class CnnEncoder(nn.Module):
    """
    src: https://github.com/baosenguo/Kaggle-MoA-2nd-Place-Solution/blob/main/training/1d-cnn-train.ipynb
    """

    def __init__(self, num_features=2048, num_targets=128, hidden_size=512, dropout=0.3):
        super().__init__()
        cha_1 = 64
        cha_2 = 128
        cha_3 = 128

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dense1 = nn.utils.weight_norm(
            nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(dropout*0.9)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            cha_1, cha_2, kernel_size=5, stride=1, padding=2,  bias=False), dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(dropout*0.8)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(dropout*0.6)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(
            cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(dropout*0.5)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(
            cha_2, cha_3, kernel_size=5, stride=1, padding=2, bias=True), dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0], self.cha_1,
                      self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x = x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


# class MultiIndexModelCnn(BaseModel):
#     def __init__(self,
#                  cols_config_path="", emb_feat_dim=32, mask_feat_ratio=0, dropout=0.3, hidden_size=128, mask_row_ratio=0,
#                  temporal_aggregator_type="TemporalTransformerAggregator", temporal_aggregator_args={},
#                  ):
#         super().__init__()

#         input_dim, num_idxs, cat_dims, cat_idxs, _ = parse_cols_config(
#             cols_config_path)
#         self.hidden_size = hidden_size
#         self.mask_row_ratio = mask_row_ratio
#         self.embedder = FixedEmbedder(
#             input_dim, emb_feat_dim, num_idxs, cat_idxs, cat_dims, mask_feat_ratio)
#         self.row_encoder = CnnEncoder(
#             self.embedder.post_embed_dim,
#             num_targets=self.hidden_size,
#             hidden_size=self.hidden_size*6,
#             dropout=dropout*0.75
#         )
#         self.rows_aggregator = nn.Sequential(
#             nn.Linear(hidden_size*49, hidden_size*6),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout),

#             nn.Linear(hidden_size*6, hidden_size*4),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout*(2/3)),

#             nn.Linear(hidden_size*4, hidden_size*2),
#             nn.LeakyReLU(),
#             nn.Dropout(dropout*0.5),

#             nn.Linear(hidden_size*2, hidden_size*1),
#             nn.LeakyReLU(),
#         )
#         self.temporal_aggregator = eval(
#             f"{temporal_aggregator_type}")(**temporal_aggregator_args)
#         self.classifier = nn.Sequential(
#             nn.Linear(temporal_aggregator_args["hidden_size"], 49),
#         )

#     def forward(self, x):
#         batch_indices, x = x
#         batch_size = int(torch.max(batch_indices) + 1)
#         if self.mask_row_ratio > 0 and self.training:
#             mask = torch.rand(len(x)) > self.mask_row_ratio
#             x = x[mask]
#             batch_indices = batch_indices[mask]
#         dts, shoptags, txn_amt = x[:, 0], x[:, 1], x[:, -1]

#         x = self.embedder(x[:, 1:])
#         rows_emb = self.row_encoder(x) * txn_amt.view(-1, 1)
#         _x = torch.zeros((batch_size, 24, 49, self.hidden_size)).to(
#             x.get_device())
#         _x[batch_indices.long(), dts.long(), shoptags.long()] = rows_emb
#         x = self.rows_aggregator(_x.view(batch_size, 24, -1))
#         x = self.temporal_aggregator(x)
#         x = self.classifier(x)
#         return x


if __name__ == "__main__":
    pass
    # model = SelfAttenNN('../data/preprocessed/v1/column_config_generated.yml')
    # data = torch.zeros((32, 200, 52))
    # print(model(data))

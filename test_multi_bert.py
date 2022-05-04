import torch
import torch.nn as nn
from transformers import AutoModel


class BertBaseWithVisual(nn.Module):
    def __init__(self, bert_path='bert-base-chinese', attr_path=''):
        super().__init__()

        self.img_hidden_size = 768
        self.img_encoder = nn.Linear(2048, 13*self.img_hidden_size)
        self.img_classifiers = nn.ModuleList([
            nn.Linear(self.img_hidden_size, num_classes)
            for num_classes in attr_config['attr_num_classes'].values()
        ])

        self.text_encoder = AutoModel.from_pretrained(bert_path)
    
    def forward(self, x):
        if len(x) == 2:
            text_ids, img_feature = x
            text_mask = None
        else:
            text_ids, text_mask, img_feature = x

        
        img_feature = self.img_encoder(img_feature).view(-1, 13, 1024)

        embedding_output = self.text_encoder.embeddings(
            input_ids=text_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )

        embedding_output = torch.cat((embedding_output, img_feature), dim=1)




    

        
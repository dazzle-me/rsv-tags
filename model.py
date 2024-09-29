import os

import pandas as pd
import numpy as np 

import torch

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import gc
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


num_target_level0, num_target_level1 = 20, 19


class CustomModel(nn.Module):
    def __init__(self, model, fc_dropout = [0.3], nn_dp = 0., lns = 1e-07, config_path=None, pretrained=False, num_feat=1000):
        super().__init__()

        if config_path is None:
            self.config = AutoConfig.from_pretrained(model)
        else:
            self.config = torch.load(config_path)

        self.num_labels = num_target_level0 + num_target_level1
        self.config.update(
            {
                'hidden_dropout_prob': nn_dp,
                "output_hidden_states": True,
                'layer_norm_eps': lns,
                "num_labels": 1,
            }
        )

        if pretrained:
            self.model = AutoModel.from_pretrained(model, config=self.config)
        else:
            self.model = AutoModel(self.config)

        self.num_dropout = len(fc_dropout)
        self.fc_dropout0 = nn.Dropout(fc_dropout[0])
        self.fc_dropout1 = nn.Dropout(fc_dropout[1] if len(fc_dropout) > 1 else 0)
        self.fc_dropout2 = nn.Dropout(fc_dropout[2] if len(fc_dropout) > 2 else 0)
        self.fc_dropout3 = nn.Dropout(fc_dropout[3] if len(fc_dropout) > 3 else 0)
        self.fc_dropout4 = nn.Dropout(fc_dropout[4] if len(fc_dropout) > 4 else 0)

        self.l0 = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(num_feat, 256),
                    nn.BatchNorm1d(256),
                    nn.SiLU(inplace=True),
                    nn.Linear(256, 64))
        
        self.fc = nn.Linear(self.config.hidden_size + 64, self.num_labels)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0][:,0,:].squeeze(1)
        return last_hidden_states


    def forward(self, inputs, feat):
        feature = self.feature(inputs)
        feat = self.l0(feat)
        feature = torch.concat([feature, feat], dim=1)
        output_list = []
        output0 = self.fc(self.fc_dropout0(feature))
        output1 = self.fc(self.fc_dropout1(feature))
        output2 = self.fc(self.fc_dropout2(feature))
        output3 = self.fc(self.fc_dropout3(feature))
        output4 = self.fc(self.fc_dropout4(feature))

        output_list = [output0, output1, output2, output3, output4]
        return output_list[:self.num_dropout]
    


class RutubeDatasetTest(Dataset):
    def __init__(self, video_df, videos_ids, title, description, tokenizer, max_len = 512):
        self.video_df = video_df
        self.videos_ids = videos_ids
        self.title = title
        self.description = description
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        video_id = self.videos_ids[idx]
        title, description = self.title[idx], self.description[idx]
        text = f'{title}. Описание - {description}'

        tok = self.tokenizer(text, max_length=self.max_len, truncation=True)

        video_feat = self.video_df.loc[video_id].values.astype('float') #[1:]
        
        return tok, video_feat
    

class Collate:
    def __init__(self, tokenizer, is_train = False):
        self.tokenizer = tokenizer
        self.is_train = is_train
    def __call__(self, batch):

        inputs = [sample[0] for sample in batch]
        feat = [sample[1] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids['input_ids']) for ids in inputs])
        # add padding
        inputs_dict = dict()
        inputs_dict["attention_mask"] = [s['attention_mask'] + (batch_max - len(s['attention_mask'])) * [0] for s in inputs]
        inputs_dict["input_ids"] = [s['input_ids'] + (batch_max - len(s['attention_mask'])) * [0] for s in inputs]
        # convert to tensors
        inputs_dict["attention_mask"] = torch.tensor(inputs_dict["attention_mask"], dtype=torch.long)
        inputs_dict["input_ids"] = torch.tensor(inputs_dict["input_ids"], dtype=torch.long)

        feat = torch.tensor(np.array(feat), dtype=torch.float)

        return inputs_dict, feat
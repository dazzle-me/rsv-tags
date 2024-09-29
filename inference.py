import os

import pandas as pd
import numpy as np 
import ffmpeg

import timm
import torch

from PIL import Image

from tqdm import tqdm

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

from model import CustomModel, RutubeDatasetTest, Collate


DF_PATH = "baseline/train_data_categories.csv" # датафрейм в формате соревнования
VIDEO_DF_PATH = 'video_feat.csv' # датафрейм с фичами из видео
OUT_PATH = 'submission.csv' # результат
tags = pd.read_csv('/home/ssd/tag_video/baseline/IAB_tags.csv')

MODEL_WEIGHTS = 'weights'


df = pd.read_csv(DF_PATH)
df_video = pd.read_csv(VIDEO_DF_PATH, index_col='video_id')



print('Predicting...')

model_name = "ai-forever/ruRoberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

models = []
for fold in [0, 1, 2, 3, 4]:
    model = CustomModel(model_name, pretrained=True).eval().cuda()
    ckp = f'{MODEL_WEIGHTS}/model_1_feat_fold_{fold}.pt'
    checkpoint = torch.load(ckp, map_location='cpu')
    model.load_state_dict(checkpoint)
    models.append(model)

collate_fn = Collate(tokenizer)
params_valid = {'batch_size': 4, 'shuffle': False, 'drop_last': False, 'num_workers': 1}
valid_dataloader = DataLoader(RutubeDatasetTest(df_video, df['video_id'], df['title'], df['description'], tokenizer), 
                                  collate_fn  = collate_fn, **params_valid)




preds = []

len_loader = len(valid_dataloader)
tk0 = tqdm(enumerate(valid_dataloader), total = len_loader)

with torch.no_grad():
    for batch_number,  (inputs, feat)  in tk0:
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        feat = feat.cuda()

        y_preds_list = []
        with torch.cuda.amp.autocast():
            for model in models:
                y_preds_list.extend(model(inputs, feat))
        # break

        y_preds = sum(y_preds_list) / len(y_preds_list)

        preds += [y_preds.sigmoid().to('cpu').numpy()]

preds = np.concatenate(preds)


num_target_level0, num_target_level1 = 20, 19


TH = 0.25

dict_target_level_0_inv = {0: 'Бизнес и финансы',
                        1: 'Дом и сад',
                        2: 'Еда и напитки',
                        3: 'Изобразительное искусство',
                        4: 'Карьера',
                        5: 'Личные финансы',
                        6: 'Массовая культура',
                        7: 'Музыка и аудио',
                        8: 'Наука',
                        9: 'Новости и политика',
                        10: 'Образование',
                        11: 'Путешествия',
                        12: 'Религия и духовность',
                        13: 'Семья и отношения',
                        14: 'События и достопримечательности',
                        15: 'Спорт',
                        16: 'Стиль и красота',
                        17: 'Транспорт',
                        18: 'Фильмы и анимация',
                        19: 'Хобби и интересы'}


dict_target_level_1_inv = {0: 'Автогонки',
                        1: 'Астрология',
                        2: 'Борьба',
                        3: 'Декоративно-прикладное искусство',
                        4: 'Дизайн интерьера',
                        5: 'Документальные фильмы',
                        6: 'Исторические места и достопримечательности',
                        7: 'Комедия и стендап',
                        8: 'Комедия и стендап (Музыка и аудио)',
                        9: 'Концерты и музыкальные мероприятия',
                        10: 'Кулинария',
                        11: 'Онлайн-образование',
                        12: 'Отношения знаменитостей',
                        13: 'Поп-музыка',
                        14: 'Рыбалка',
                        15: 'Семейные и детские фильмы',
                        16: 'Скандалы знаменитостей',
                        17: 'Спортивные события',
                        18: 'Юмор и сатира'}



list_predicts = []
for pr in preds:
    ind = np.where(pr > TH)[0]
    tmp_pred = []
    if len(ind):
        for i in ind:
            if i >= num_target_level0:
                tmp_pred += [': '.join(tags[tags.iloc[:, 1] == dict_target_level_1_inv[i - num_target_level0]].iloc[0].tolist()[:2])]
            else:
                tmp_pred += [ dict_target_level_0_inv[i]]
    list_predicts += [str(tmp_pred)]

pred_submission = pd.DataFrame()
pred_submission['video_id'] = df['video_id']
pred_submission['predicted_tags'] = list_predicts

pred_submission.to_csv(OUT_PATH, index_label=0)
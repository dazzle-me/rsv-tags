import os

import pandas as pd
import numpy as np 
import ffmpeg

import timm
import torch

from PIL import Image

from tqdm import tqdm



DF_PATH = "baseline/train_data_categories.csv" # датафрейм в формате соревнования
VIDEO_PATH = "videos_2" # папка с видео
OUT_PATH = 'video_feat.csv' # результат с фичами из видео

FRAMES_PATH = "frames" # сохраняем сюда кадры
MODEL_NAME = 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k'


df = pd.read_csv(DF_PATH)

print('Extracting frames from videos...')

os.makedirs('{FRAMES_PATH}', exist_ok=True)
for v in tqdm(df['video_id'].values):
    os.makedirs(f'{FRAMES_PATH}/{v}', exist_ok=True)
    
    fn = f'{VIDEO_PATH}/{v}.mp4'
    probe = ffmpeg.probe(fn)
    time = int(float(probe['streams'][0]['duration']))
    width = probe['streams'][0]['width']

    parts = min(100, time)

    intervals = time // parts
    intervals = int(intervals)
    interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
    i = 0

    for item in interval_list:
        (
            ffmpeg
            .input(fn, ss=item[0])
            .filter('scale', width, -1)
            .output(f'{FRAMES_PATH}/{v}/img_{item[0]}.jpg', vframes=1, loglevel="quiet")
            .run(overwrite_output=True)
        )
        i += 1


print('Extracting features from frames...')


model = timm.create_model(MODEL_NAME, pretrained=True)

data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)
model = model.cuda().eval()

res_df = []

for v in tqdm(df['video_id'].values):
    files = sorted(os.listdir(f'{FRAMES_PATH}/{v}'))
    v_preds = []
    for fn in files:
        image = Image.open(f'{FRAMES_PATH}/{v}/{fn}')
        
        with torch.no_grad():
            image_tensor = transform(image).cuda()
            output = model(image_tensor.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        v_preds.append(probabilities)

    v_preds = np.asarray(v_preds)
    v_preds = v_preds.max(axis=0)

    res = {'video_id': v}
    for i in range(len(v_preds)):
        res[f'vfeat_{i}'] = v_preds[i]

    res_df.append(res)


res_df = pd.DataFrame(res_df)
res_df.to_csv(OUT_PATH)
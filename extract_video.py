import os
import argparse

import pandas as pd
import numpy as np
import ffmpeg

import timm
import torch

from PIL import Image
from tqdm import tqdm


def main(df_path, video_path, out_path, frames_path, model_name):
    # Считываем датафрейм с путями к видеофайлам и категориями
    df = pd.read_csv(df_path)

    print('Извлечение кадров из видео...')

    # Создаем папку для хранения кадров, если её ещё нет
    os.makedirs(frames_path, exist_ok=True)
    for v in tqdm(df['video_id'].values):
        os.makedirs(f'{frames_path}/{v}', exist_ok=True)

        # Получаем информацию о видеофайле через ffmpeg
        fn = f'{video_path}/{v}.mp4'
        probe = ffmpeg.probe(fn)
        time = int(float(probe['streams'][0]['duration']))
        width = probe['streams'][0]['width']

        # Определяем количество частей, на которые разбиваем видео, и интервалы кадров
        parts = min(100, time)
        intervals = time // parts
        interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]

        # Извлекаем кадры в указанные интервалы и сохраняем их в папку
        for item in interval_list:
            (
                ffmpeg
                .input(fn, ss=item[0])
                .filter('scale', width, -1)
                .output(f'{frames_path}/{v}/img_{item[0]}.jpg', vframes=1, loglevel="quiet")
                .run(overwrite_output=True)
            )

    print('Извлечение признаков из кадров...')

    # Загружаем предобученную модель для извлечения фичей
    model = timm.create_model(model_name, pretrained=True)
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
    transform = timm.data.create_transform(**data_cfg)
    model = model.cuda().eval()  # Переносим модель на GPU и переводим в режим оценки

    res_df = []

    # Проходим по каждому видео и извлекаем фичи из каждого кадра
    for v in tqdm(df['video_id'].values):
        files = sorted(os.listdir(f'{frames_path}/{v}'))
        v_preds = []
        for fn in files:
            # Открываем изображение и применяем необходимые преобразования
            image = Image.open(f'{frames_path}/{v}/{fn}')

            # Извлечение признаков с использованием модели
            with torch.no_grad():
                image_tensor = transform(image).cuda()
                output = model(image_tensor.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
            v_preds.append(probabilities)

        # Объединяем результаты и находим максимальное значение по каждому признаку
        v_preds = np.asarray(v_preds)
        v_preds = v_preds.max(axis=0)

        # Создаем словарь с результатами для текущего видео
        res = {'video_id': v}
        for i in range(len(v_preds)):
            res[f'vfeat_{i}'] = v_preds[i]

        res_df.append(res)

    # Сохраняем итоговый DataFrame с признаками в CSV файл
    res_df = pd.DataFrame(res_df)
    res_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    # Инициализируем argparse и добавляем аргументы командной строки
    parser = argparse.ArgumentParser(description="Скрипт для извлечения признаков из видеокадров с использованием предобученной модели.")
    
    # Добавляем аргумент для пути к датафрейму с категориями
    parser.add_argument('--df_path', type=str, default="baseline/train_data_categories.csv",
                        help="Путь к датафрейму с категориями и ID видео.")
    
    # Добавляем аргумент для директории с видеофайлами
    parser.add_argument('--video_path', type=str, default="videos_2", 
                        help="Папка, содержащая видеофайлы.")
    
    # Добавляем аргумент для пути, куда сохранить CSV файл с признаками
    parser.add_argument('--out_path', type=str, default='video_feat.csv', 
                        help="Путь для сохранения итогового CSV файла с извлеченными признаками.")
    
    # Добавляем аргумент для директории, куда сохраняются извлеченные кадры
    parser.add_argument('--frames_path', type=str, default='frames', 
                        help="Папка для хранения извлеченных кадров.")
    
    # Добавляем аргумент для имени предобученной модели
    parser.add_argument('--model_name', type=str, default='eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', 
                        help="Имя предобученной модели для извлечения признаков.")

    # Парсим аргументы командной строки
    args = parser.parse_args()

    # Запускаем основную функцию с аргументами
    main(args.df_path, args.video_path, args.out_path, args.frames_path, args.model_name)

import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm

from model import CustomModel, RutubeDatasetTest, Collate


# Определение аргументов командной строки
def parse_args():
    parser = argparse.ArgumentParser(description="Скрипт для предсказания тегов видео на основе входных данных", 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--df_path', type=str, default="baseline/train_data_categories.csv", 
                        help='Путь к CSV файлу с обучающими данными')
    parser.add_argument('--video_df_path', type=str, default='video_feat.csv', 
                        help='Путь к CSV файлу с фичами видео')
    parser.add_argument('--out_path', type=str, default='submission.csv', 
                        help='Путь к выходному файлу для сохранения предсказаний')
    parser.add_argument('--model_weights', type=str, default='weights', 
                        help='Путь к папке с весами модели')
    parser.add_argument('--model_name', type=str, default="ai-forever/ruRoberta-large", 
                        help='Название или путь к модели')
    parser.add_argument('--tags_path', type=str, default='/home/ssd/tag_video/baseline/IAB_tags.csv', 
                        help='Путь к файлу с тегами IAB')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Размер батча для загрузчика данных')
    parser.add_argument('--threshold', type=float, default=0.25, 
                        help='Порог вероятности для предсказания тегов')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Количество потоков для загрузки данных')

    return parser.parse_args()


def main():
    args = parse_args()

    # Загрузка данных
    df = pd.read_csv(args.df_path)
    df_video = pd.read_csv(args.video_df_path, index_col='video_id')
    tags = pd.read_csv(args.tags_path)

    print('Загрузка и подготовка модели...')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Инициализация моделей и загрузка весов
    models = []
    for fold in [0, 1, 2, 3, 4]:
        model = CustomModel(args.model_name, pretrained=True).eval().cuda()
        ckp = f'{args.model_weights}/model_1_feat_fold_{fold}.pt'
        checkpoint = torch.load(ckp, map_location='cpu')
        model.load_state_dict(checkpoint)
        models.append(model)

    # Настройка DataLoader
    collate_fn = Collate(tokenizer)
    params_valid = {'batch_size': args.batch_size, 'shuffle': False, 'drop_last': False, 'num_workers': args.num_workers}
    valid_dataloader = DataLoader(RutubeDatasetTest(df_video, df['video_id'], df['title'], df['description'], tokenizer), 
                                  collate_fn=collate_fn, **params_valid)

    print('Предсказание...')

    # Предсказание
    preds = []
    len_loader = len(valid_dataloader)
    tk0 = tqdm(enumerate(valid_dataloader), total=len_loader)

    with torch.no_grad():
        for batch_number, (inputs, feat) in tk0:
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            feat = feat.cuda()

            y_preds_list = []
            with torch.cuda.amp.autocast():
                for model in models:
                    y_preds_list.extend(model(inputs, feat))

            y_preds = sum(y_preds_list) / len(y_preds_list)
            preds += [y_preds.sigmoid().to('cpu').numpy()]

    preds = np.concatenate(preds)

    # Словари для декодирования
    num_target_level0, num_target_level1 = 20, 19
    dict_target_level_0_inv = {
        0: 'Бизнес и финансы',
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
        19: 'Хобби и интересы'
    }

    dict_target_level_1_inv = {
        0: 'Автогонки',
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
        18: 'Юмор и сатира'
    }

    # Постпроцессинг предсказаний
    list_predicts = []
    for pr in preds:
        ind = np.where(pr > args.threshold)[0]
        tmp_pred = []
        if len(ind):
            for i in ind:
                if i >= num_target_level0:
                    tmp_pred += [': '.join(tags[tags.iloc[:, 1] == dict_target_level_1_inv[i - num_target_level0]].iloc[0].tolist()[:2])]
                else:
                    tmp_pred += [dict_target_level_0_inv[i]]
        list_predicts += [str(tmp_pred)]

    pred_submission = pd.DataFrame()
    pred_submission['video_id'] = df['video_id']
    pred_submission['predicted_tags'] = list_predicts

    # Сохранение предсказаний
    pred_submission.to_csv(args.out_path, index_label=0)
    print(f'Предсказания сохранены в {args.out_path}')


if __name__ == "__main__":
    main()

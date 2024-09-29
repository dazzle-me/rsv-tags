# Извлечение тегов из видео.

## Окружение
Ожидается что в окружении стоит pytorch и стандартные ML библиотеки. Для NLP моделей мы использовали hf
```
pip install transformers --upgrade
pip install accelerate
```
Для инференса аудио-фичей
```
pip install whisper
```

Для инференса видео-фичей
```
pip install timm
```

Так же понадобится `ffmpeg` 
```
apt-get install ffmpeg
```

## Инференс

Наше решение использует признаки из всех модальностей - текст (название + описание), аудио (whisper), видео (imagenet classes). 

Для извлечения аудио и транскрибирования (**в финальной модели не используется**) - 
```
python3 extract_audio.py --csv_path baseline/train_data_categories.csv \
                         --video_dir videos/ \
                         --audio_dir audios/ \
                         --audio_duration 60 \
                         --transcribe_dir audio_text/
```

Для извлечения кадров из видео и сохранения вероятностей классов из ImageNet - 
```
python3 extract_video.py --df_path baseline/train_data_categories.csv \
                         --video_path videos/ \
                         --frames_path frames/ \
                         --out_path video_feat.csv
```

После того как вы скопировали репозиторий локально, веса для модели необходимо поместить в папку `weights` в корне проекта для того чтобы `inference.py` корректно отработал. 

Веса для натренированной модели находятся [здесь](https://github.com/dazzle-me/rsv-tags/releases/tag/weights). 

После извлечения признаков мы запускаем инференс, предсказываем теги - 
```
python3 inference.py --df_path baseline/train_data_categories.csv \
                     --video_df_path video_feat.csv \
                     --out_path submission.csv \
                     --model_weights weights \
                     --tags_path baseline/IAB_tags.csv
```

Сгенерированный `submission.csv` можно сабмитить на лидерборд, этот сабмит получает скор `0.704`.

Мы так же предоставляем возможность воспроизвести веса наших обученных моделей, соответствующий код находится в `train.ipynb`, однако там придется самостоятельно поменять пути до данных.

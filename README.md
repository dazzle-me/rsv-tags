# Извлечение тегов из видео.

Наше решение использует признаки из всех модальностей - текст (название + описание), аудио (whisper), видео (imagenet classes). 

Для извлечения аудио и транскрибирования - 
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

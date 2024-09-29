# Извлечение тегов из видео.

Наше решение использует признаки из всех модальностей - текст (название + описание), аудио (whisper), видео (imagenet classes). 

Для извлечения аудио и транскрибирования - ```python3 extract_audio.py```
Для извлечения кадров из видео и сохранения вероятностей классов из ImageNet - ```python3 extract_video.py```
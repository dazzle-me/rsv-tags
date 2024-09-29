import pandas as pd
import os
import numpy as np
from pathlib import Path
import subprocess
import tqdm
import whisper
import argparse

def main(args):
    ## Прочитаем датафрейм
    df = pd.read_csv(args.csv_path)
    
    ## Убедимся, что все видео заканчиваются на .mp4
    if not np.all([str(x).endswith('.mp4') for x in os.listdir(args.video_dir)]):
        print("Некоторые видео не имеют расширения .mp4. Проверьте корректность файлов.")
        return

    ## Извлечем первую минуту аудио из каждого видео и сохраним в отдельную папку
    audio_duration = args.audio_duration
    video_dir = Path(args.video_dir)
    audio_dir = Path(f'{args.audio_dir}')
    audio_dir.mkdir(exist_ok=True)

    for vid in tqdm.tqdm(df['video_id']):
        ### Пояснение команды:
        # -i input_video.mp4: Указывает на входной видеофайл.
        # -t 60: Ограничивает длительность извлекаемого аудио до 60 секунд (1 минута).
        # -q:a 0: Устанавливает наилучшее качество аудио (опционально, для MP3).
        # -map a: Извлекает только аудио-дорожку, исключая видео.
        # output_audio.mp3: Указывает на выходное имя файла и формат (например, .wav, .aac и т.д.).
        cmd = f'ffmpeg -i {video_dir / vid}.mp4 -t {audio_duration} -q:a 0 -map a {audio_dir / vid}.mp3'
        subprocess.call(cmd, shell=True)

    ## Загрузим модель whisper
    model = whisper.load_model(args.model_name, device=args.device)

    ## Извлечем текст из каждой аудиозаписи и сохраним в отдельную папку
    transcribe_dir = Path(f'{args.transcribe_dir}')
    transcribe_dir.mkdir(exist_ok=True)
    for aud in tqdm.tqdm(df['video_id']):
        result = model.transcribe(str(audio_dir / f"{aud}.mp3"))
        with open(transcribe_dir / f"{aud}.txt", 'w') as f:
            f.write(result['text'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт для извлечения и транскрибирования аудио из видеофайлов")
    
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Путь к CSV файлу, содержащему информацию о видеофайлах.')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Путь к директории с видеофайлами.')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Путь к директории для сохранения извлечённых аудиофайлов.')
    parser.add_argument('--audio_duration', type=int, default=60,
                        help='Длительность извлекаемого аудио в секундах (по умолчанию 60).')
    parser.add_argument('--model_name', type=str, default='small',
                        help='Имя модели Whisper для транскрибирования (по умолчанию "small").')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Устройство для вычислений ("cuda:0" или "cpu").')
    parser.add_argument('--transcribe_dir', type=str, required=True,
                        help='Путь к директории для сохранения транскрибированных текстов.')

    args = parser.parse_args()
    main(args)

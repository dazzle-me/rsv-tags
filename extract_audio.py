import pandas as pd
import os
import numpy as np
from pathlib import Path
import subprocess
import tqdm
import whisper

if __name__ == '__main__':
    ## Прочитаем датафрейм
    df = pd.read_csv('/home/ssd/tag_video/baseline/train_data_categories.csv')
    
    ## убедимся что все видео заканчиваются на .mp4
    print(np.all([str(x).endswith('.mp4') for x in os.listdir('/home/ssd/tag_video/videos_2')]))

    ## извлечем первую минуту аудио из каждого видео и сложим в отдельную папку
    audio_duration = 60
    video_dir = Path('/home/ssd/tag_video/videos_2')
    audio_dir = Path(f'/home/ssd/tag_video/audio_{audio_duration}')
    audio_dir.mkdir(exist_ok=True)

    for vid in tqdm.tqdm(df['video_id']):
        ### Explanation:
        # -i input_video.mp4: Specifies the input video file.
        # -t 60: Limits the duration of the extracted audio to 60 seconds (1 minute).
        # -q:a 0: Sets the audio quality to the best possible (optional, for MP3 output).
        # -map a: Maps only the audio stream, excluding video.
        # output_audio.mp3: Specifies the output file name and format. You can change the extension (e.g., .wav, .aac) based on your desired format.
        cmd = f'ffmpeg -i {video_dir / vid}.mp4 -t {audio_duration} -q:a 0 -map a {audio_dir / vid}.mp3',
        subprocess.call(
            cmd, shell=True,
        )

    ## загрузим whisper
    model = whisper.load_model('small', device='cuda:0')

    ## извлечем текст из каждой аудиозаписи и опять сложим в отдельную папку
    transcribe_dir = Path(f'/home/ssd/tag_video/transcribe_{audio_duration}')
    transcribe_dir.mkdir(exist_ok=True)
    for aud in tqdm.tqdm(df['video_id']):
        result = model.transcribe(str(audio_dir / f"{aud}.mp3"))
        with open(transcribe_dir / f"{aud}.txt", 'w') as f:
            f.write(result['text'])

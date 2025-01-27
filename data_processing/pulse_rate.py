import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from read_edf import get_channel

CHANNEL_ID = 'Mic'
DIR = '/var/data/apnea/edf/'
OUT_DIR = '/var/data/apnea/mic/'

file_names = os.listdir(DIR)

for file_name in tqdm(file_names):
    print(f'{file_name} processing')

    file_path = DIR + file_name
    output_file = OUT_DIR + file_name.replace('.edf', '.wav')
    #output_txt_file = OUT_TXT_DIR + file_name.replace('.edf', '.txt')
    
    data, info = get_channel(file_path, channel_id=CHANNEL_ID)
    sfreq = int(info['sfreq'])

    new_sample_rate = 16000

    data = librosa.resample(data, orig_sr=sfreq, target_sr=new_sample_rate)

    max_value = np.max(np.abs(data))
    scaled_data = data / max_value

    sf.write(output_file, scaled_data, new_sample_rate)
    print(f"Аудио сохранено в файл: {output_file}")
    
print("FINISH!!!")
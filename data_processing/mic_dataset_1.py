# %%
import mne
import os
import time
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from tqdm import tqdm   
import xml.etree.ElementTree as ET
from scipy.io import wavfile
from scipy.signal import find_peaks
import warnings

from read_rml import get_attrubuts

# %%
MIC_DIR = '/var/data/apnea/edf_channels/mic/'
RML_DIR = '/var/data/apnea/rml/'

mic_files = sorted(os.listdir(MIC_DIR), key=lambda x: 
                    (int(x.split('-')[0]), int(x.split('-')[1][-8])))

rml_files = sorted(os.listdir(RML_DIR), key=lambda x: 
                    (int(x.split('-')[0]), int(x.split('-')[1][-8])))

# %%
WINDOW_DURATION = 5 
OUT_DIR = '/var/data/apnea/mic_dataset_5s/1/'
MAX_WINDOWS = 1500

# %%
for rml_file in tqdm(rml_files):
    rml_path = RML_DIR + rml_file
    name = rml_file.replace('.rml', '')
    count_files = 0
    print(rml_file)

    apnea_attributs = get_attrubuts(rml_path)

    for mic_file in mic_files:
        if name in mic_file:
            mic_path = MIC_DIR + mic_file 
        
            mic_audio, sr = librosa.load(mic_path, sr=None)
            
            len_sec = int(len(mic_audio)/sr)
            labels = np.zeros(len_sec)
        
            flag = False
            for elem in apnea_attributs:
                start = int(float(elem['Start'])) - count_files*60*60
                end = int(start + float(elem['Duration'])) 
                #print(count_files, start, end)

                if start < 0:
                    continue
                elif start < len(mic_audio)/sr:
                    flag = True
                    labels[start:end] = 1

                    start_sample = int(start * sr)
                    end_sample = int(min(len(mic_audio), end * sr))
                    apnea_segment = mic_audio[start_sample:end_sample]

                    window_counter = 0
                    for i in range(0, len(apnea_segment), WINDOW_DURATION * sr):
                        window_start = i
                        window_end = min(i + WINDOW_DURATION * sr, len(apnea_segment))
                        window_audio = apnea_segment[window_start:window_end]

                        if len(window_audio) == WINDOW_DURATION * sr:
                            output_filename = f"{OUT_DIR}/{name}_{window_counter}.wav"
                            sf.write(output_filename, window_audio, sr)
                            window_counter += 1
                else:
                    count_files += 1
                    #print('-'*50)
                    break


print("FINISH")
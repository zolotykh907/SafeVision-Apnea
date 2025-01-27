import mne
import os
import time
import random
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


warnings.filterwarnings("ignore")
RML_DIR = '/var/data/apnea/rml/'
PULSE_DIR = '/var/data/apnea/pulse_rate/'
TRACHEAL_DIR = '/var/data/apnea/tracheal/'
OUT_DIR = '/var/data/apnea/spec_img_dataset/'
OUT_PULSE_DIR = '/var/data/apnea/spec_img_dataset/pulse/'
OUT_TRACHEAL_DIR = '/var/data/apnea/spec_img_dataset/tracheal/'

rml_files = sorted(os.listdir(RML_DIR), key=lambda x: 
                   int(x.split('-')[0]))
pulse_files = sorted(os.listdir(PULSE_DIR), key=lambda x: 
                    (int(x.split('-')[0]), int(x.split('-')[1][-8])))
tracheal_files = sorted(os.listdir(TRACHEAL_DIR), key=lambda x: 
                    (int(x.split('-')[0]), int(x.split('-')[1][-8])))


def extract_mfcc(audio, sr, n_mfcc=13):
                return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

def save_mfcc_as_image(mfcc, sr, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_labels(labels, save_path):
    with open(save_path, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")


for rml_file in tqdm(rml_files):
    rml_path = RML_DIR + rml_file
    name = rml_file.replace('.rml', '')
    print(rml_file)

    apnea_attributs = get_attrubuts(rml_path)
    #print(len(apnea_attributs))

    count_files = 0

    for pulse_file in pulse_files:
        if name in pulse_file:
            pulse_path = PULSE_DIR + pulse_file 
            tracheal_path = TRACHEAL_DIR + pulse_file

            pulse_audio, sr = librosa.load(pulse_path, sr=None)
            tracheal_audio, _ = librosa.load(tracheal_path, sr=sr)
            
            len_sec = int(len(pulse_audio)/sr)
            #print(len_sec)

            labels = np.zeros(len_sec)
            
            flag = False
            for elem in apnea_attributs:
                start = int(float(elem['Start'])) - count_files*60*60
                end = int(start + float(elem['Duration'])) 
                #print(count_files, start, end)

                if start < 0:
                    continue
                elif start < len(pulse_audio)/sr:
                    flag = True
                    labels[start:end] = 1
                else:
                    count_files += 1
                    #print('-'*50)
                    break
            
            if flag:
                num_seconds = 1
                window_size = sr * num_seconds

                num_windows = len(pulse_audio) // window_size

                windows_pulse = np.array_split(pulse_audio[:num_windows * window_size], num_windows)
                windows_tracheal = np.array_split(tracheal_audio[:num_windows * window_size], num_windows)

                num_samples = 10
                selected_windows_pulse = []
                selected_windows_tracheal = []
                selected_labels = []
                
                one_flag = False

                for j, label in enumerate(labels):
                    if not one_flag:
                        if label == 1:
                            one_flag = True
                            start_id = max(0, j-num_samples)
                            end_id = min(num_windows, start_id+num_samples)
                            
                            for l in range(j, num_windows):
                                if labels[l] == 0:
                                    end_id = min(num_windows, l+num_samples)
                                    break
                            #print(start_id, end_id, l)
                            
                            for k in range(start_id, end_id):
                                selected_windows_pulse.append(windows_pulse[k])
                                selected_windows_tracheal.append(windows_tracheal[k])
                                selected_labels.append(labels[k])

                    if one_flag:
                        if label == 0:
                            one_flag = False

                #print('len selected windows -', len(selected_windows_pulse))
                
                mfccs_pulse = ([extract_mfcc(window, sr) for window in selected_windows_pulse])
                mfccs_tracheal = ([extract_mfcc(window, sr) for window in selected_windows_tracheal])
                print(len(mfccs_pulse))
                save_dir = OUT_DIR + pulse_file.replace('.wav', '')
                os.makedirs(save_dir, exist_ok=True)
                
                for mfcc_id, (mfcc_p, mfcc_t) in enumerate(zip(mfccs_pulse, mfccs_tracheal)):
                    save_mfcc_as_image(mfcc_p, sr, os.path.join(save_dir, f"pulse_{mfcc_id}.png"))
                    save_mfcc_as_image(mfcc_t, sr, os.path.join(save_dir, f"tracheal_{mfcc_id}.png"))

                save_labels(selected_labels, os.path.join(save_dir, "labels.txt"))
                
print("ABOBA")
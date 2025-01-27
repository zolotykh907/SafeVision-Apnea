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

import tensorflow as tf


from data_processing.read_rml import get_attrubuts

# %%
import sys
sys.path.append("/var/data/apnea/src/vggish")

# %%
import vggish_input, vggish_params, vggish_slim
from vggish.vggish_slim import define_vggish_slim, load_vggish_slim_checkpoint

checkpoint_path = "/var/data/apnea/src/vggish/vggish_model.ckpt"  
pca_params_path = "/var/data/apnea/src/vggish/vggish_pca_params.npz"

# %%
# with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
#     pool4_output = define_vggish_slim()

#     checkpoint_path = "/var/data/apnea/src/vggish/vggish_model.ckpt"
#     load_vggish_slim_checkpoint(sess, checkpoint_path)

#     for op in sess.graph.get_operations():
#         print(op.name)

# %%
def split_audio_to_windows(audio_path, window_size=1.0, step_size=0.25):
    # Загрузка аудиофайла
    wav_data, sample_rate = librosa.load(audio_path, sr=None)
    #print(wav_data)
    
    # Преобразование в моно и нормализация
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)

    # Разделение на окна
    window_samples = int(window_size * sample_rate)
    step_samples = int(step_size * sample_rate)
    num_windows = (len(wav_data) - window_samples) // step_samples + 1

    windows = []
    for i in range(num_windows):
        start = i * step_samples
        end = start + window_samples
        window = wav_data[start:end]
        windows.append(window)
    
    return windows, sample_rate

# %%
APNEA_DIR = '/var/data/apnea/mic_dataset_5s/1/'
NO_APNEA_DIR = '/var/data/apnea/mic_dataset_5s/0/'
OUT_DIR = '/var/data/apnea/mic_dataset_spec/'

apena_files = os.listdir(APNEA_DIR)
no_apnea_files = os.listdir(NO_APNEA_DIR)[:1500]

dataset = []

# %%
os.makedirs(OUT_DIR, exist_ok=True)

with tf.Graph().as_default():
    sess = tf.compat.v1.Session()
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    features_tensor = sess.graph.get_tensor_by_name("vggish/input_features:0")
    pool4_output = sess.graph.get_tensor_by_name("vggish/pool4/MaxPool:0")

    for apena_file in tqdm(apena_files):
        audio_path = APNEA_DIR + apena_file
        
        windows, sr = split_audio_to_windows(audio_path)
        spectograms = []

        for i, window in enumerate(windows):
            mel_spec = vggish_input.waveform_to_examples(window, sr)

            [pool4_output_val] = sess.run([pool4_output],
                                        feed_dict={features_tensor: mel_spec})
            pool4_output_val = np.reshape(pool4_output_val, [-1, 6 * 4 * 512])

            spectograms.append(pool4_output_val)
            
        dataset.append((spectograms, 1))

        spectograms_combined = np.stack(spectograms, axis=0)
        spectograms_combined = np.squeeze(spectograms_combined, axis=1)

        data_for_save = {'spectograms': spectograms_combined, 'label': 1}

        output_file = os.path.join(OUT_DIR, f"1/{apena_file}_combined.npy")
        np.save(output_file, data_for_save)
    
    sess.close()

# %%
with tf.Graph().as_default():
    sess = tf.compat.v1.Session()
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    features_tensor = sess.graph.get_tensor_by_name("vggish/input_features:0")
    pool4_output = sess.graph.get_tensor_by_name("vggish/pool4/MaxPool:0")

    for apena_file in tqdm(no_apnea_files):
        audio_path = NO_APNEA_DIR + apena_file
        
        windows, sr = split_audio_to_windows(audio_path)
        spectograms = []

        for i, window in enumerate(windows):
            mel_spec = vggish_input.waveform_to_examples(window, sr)

            [pool4_output_val] = sess.run([pool4_output],
                                        feed_dict={features_tensor: mel_spec})
            pool4_output_val = np.reshape(pool4_output_val, [-1, 6 * 4 * 512])

            spectograms.append(pool4_output_val)
            
        dataset.append((spectograms, 1))

        spectograms_combined = np.stack(spectograms, axis=0)
        spectograms_combined = np.squeeze(spectograms_combined, axis=1)

        data_for_save = {'spectograms': spectograms_combined, 'label': 0}

        output_file = os.path.join(OUT_DIR, f"0/{apena_file}_combined.npy")
        np.save(output_file, data_for_save)
    
    sess.close()
# %%



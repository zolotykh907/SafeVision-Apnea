import mne
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import find_peaks

import warnings
warnings.filterwarnings("ignore")

file = '/var/data/apnea/edf/00001014-100507%5B001%5D.edf'

def get_channel(filepath, channel_id='PulseRate'):
    data = mne.io.read_raw_edf(file)
    raw_data = data.get_data()
    info = data.info
    channels = data.ch_names

    selected_channel = data.copy().pick_channels([channel_id])

    selected_data = selected_channel.get_data()[0]

    return selected_data, info

# selected_data, info = get_snore_channel(file)
# print(f"Длина данных: {len(selected_data)} отсчётов")

# s = 173
# d = 15

# sfreq = int(info['sfreq'])

# start_index = int(s * sfreq)
# end_index = int((s + d) * sfreq)

# cute_data = selected_data[start_index:end_index]

# max_value = np.max(np.abs(cute_data))
# scaled_data = cute_data / max_value

# output_file = "./output_audio.wav"
# sf.write(output_file, scaled_data, sfreq)

# print(f"Аудио сохранено в файл: {output_file}")
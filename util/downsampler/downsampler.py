import sys
import os
from pathlib import Path
import re
import librosa
import soundfile
from scipy.io.wavfile import write as write_wave
import numpy as np


if len(sys.argv) != 2:
    print('''
    Error: Invalid number of args.
    Usage:
    
        python downsampler.py <target dir>
        
    Consult README.md for more info.
    Exiting...
    ''')
    exit(-1)

TARGET_DIR = sys.argv[1]
AUDIO_FILES = librosa.util.find_files(directory=TARGET_DIR, recurse=True, case_sensitive=False)
TARGET_SR = 16000
PROCESSED_SUFFIX = '16khz16bit'

for file in AUDIO_FILES:
    regex = re.compile(f'.*{PROCESSED_SUFFIX}.*', re.IGNORECASE)
    if regex.search(file):
        print(f'File [ {file} ] seems to already be downsampled. Skipping...')
        continue

    try:
        y, sr = librosa.load(path=file, sr=None)

        # convert stereo signal to mono
        if y.shape[0] == 2:
            y = np.nanmean(a=y, axis=0)

        y_16k = librosa.resample(y=y, orig_sr=sr, target_sr=TARGET_SR, res_type='kaiser_best', scale=True)

        # manually decrease bit depth - currently done by soundfile.write subtype PCM_16
        # y_16k = (y_16k / np.max(np.abs(y_16k)) * np.iinfo(np.int16).max).astype(np.int16)

        file_path = Path(file)
        new_file_name = os.path.join(str(file_path.parent), f'{file_path.stem}_{PROCESSED_SUFFIX}{file_path.suffix}')
        soundfile.write(file=new_file_name, data=y_16k, samplerate=TARGET_SR, format='WAV', subtype='PCM_16')

        # write file using scipy (result is louder than original)
        # write_wave(filename=new_file_name, rate=TARGET_SR, data=y_16k_16bit)
    except Exception:
        print(f'Error processing file [ {file} ]')
        print(Exception)

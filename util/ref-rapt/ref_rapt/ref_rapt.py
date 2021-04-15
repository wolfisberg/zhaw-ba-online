import pysptk
import numpy as np
import librosa
import os


DATA_DIR = '/home/kaspar/Glacier/data-zhaw-ba/unzipped/SPEECH DATA/MALE'
REF_RAPT_DIR = 'REF_RAPT'
REF_ORIGINAL_DIR = 'REF'
SAMPLING_RATE = 16000
# if not os.path.exists(REF_RAPT_DIR):
#     os.makedirs(REF_RAPT_DIR)


def get_wav_files():
    return librosa.util.find_files(DATA_DIR)


def pitch_estimation_ref_rapt(x, fs, hop_size=256):
    f0 = pysptk.sptk.rapt(x/np.max(np.abs(x))*(2**15-1), fs, hop_size)
    return f0


def get_ref_rapt_file_path(file_path):
    path_list = os.path.normpath(file_path).split(os.sep)
    path_list[0] = '/' if path_list[0] == '' else exit(-1)
    ref_rapt_dir_path = os.path.join(
        *path_list[:-3], REF_RAPT_DIR, path_list[-2])
    os.makedirs(name=ref_rapt_dir_path, exist_ok=True)
    file_name = os.path.splitext(path_list[-1])[0]
    return os.path.join(ref_rapt_dir_path, 'ref_rapt_{}.f0'.format(file_name[4:]))


def get_ref_original_file_path(file_path):
    path_list = os.path.normpath(file_path).split(os.sep)
    if path_list[0] == '':
        path_list[0] = '/'
    else:
        print('Error detecting path style.')
        exit(-1)
    ref_original_dir_path = os.path.join(
        *path_list[:-3], REF_ORIGINAL_DIR, path_list[-2])
    file_name = os.path.splitext(path_list[-1])[0]
    return os.path.join(ref_original_dir_path, 'ref_{}.f0'.format(file_name[4:]))


def load_ref_original(file_path):
    f0_original_path = get_ref_original_file_path(file_path)
    if not os.path.exists(f0_original_path):
        print('Error, f0 file not found for {}'.format(f0_original_path))
        exit(-1)
    f0_original = np.genfromtxt(f0_original_path, delimiter=' ').T[0]
    # Padding pitch ref data
    # pitch_refdata = np.pad(f0_original.T[0], (6, 0), 'constant')
    return f0_original


# def interpolate_ref_original(t, f0):


files = get_wav_files()
file = files[0]
ref_rapt_path = get_ref_rapt_file_path(file)
# raw, sr = librosa.load(path=file_path, sr=SAMPLING_RATE, mono=True)
f0_original = load_ref_original(file)
# f0_rapt = pitch_estimation_ref_rapt(raw, sr)

print(ref_rapt_path)

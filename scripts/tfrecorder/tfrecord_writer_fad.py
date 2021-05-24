import tensorflow as tf
import numpy as np
import librosa
from librosa import display
import os
import math
import pathlib
import matplotlib.pyplot as plt
from scipy import interpolate


_FILES_PER_SHARD = 2000
_BASE_PATH = os.path.join('/' 'home', 'kaspar', 'Glacier', 'data-zhaw-ba', 'unzipped', 'fad')
_SHARD_BASE_NAME = 'shard_fad_'
_SAMPLE_RATE = 16000

length_differences = []


def main():
    create_tfrecords_from_directory(_BASE_PATH)


def create_tfrecords_from_directory(dir):
    data_dir = 'tt'
    data_dir_path = _BASE_PATH

    wav_files = librosa.util.find_files(directory=data_dir_path, ext=['wav'], recurse=False, case_sensitive=False)
    number_of_shards = math.ceil(len(wav_files) / _FILES_PER_SHARD)

    for shard_number in range(number_of_shards):
        shard_name = f'{_SHARD_BASE_NAME}_{data_dir}_{str(shard_number).rjust(4, "0")}.tfrecord'
        shard_path = os.path.join(_BASE_PATH, 'tfrecords', data_dir, shard_name)

        with tf.io.TFRecordWriter(shard_path) as out:
            lowerIndex = shard_number * _FILES_PER_SHARD
            upperIndex = (shard_number + 1) * _FILES_PER_SHARD

            for fileIndex in range(lowerIndex, upperIndex if upperIndex <= len(wav_files) else len(wav_files)):
                file_path = wav_files[fileIndex]
                y, sr = librosa.load(file_path, sr=_SAMPLE_RATE, mono=True)

                file_name_stem = os.path.splitext(os.path.basename(file_path))[0]
                ref_matches = list(pathlib.Path(_BASE_PATH).rglob(f'*{file_name_stem}*.npy'))

                if len(ref_matches) != 1:
                    print(f'Cannot find ref file for [ {file_path} ], skipping...')
                    continue

                ref_data = np.load(str(ref_matches[0]))
                ref_f0 = [x[1] for x in ref_data]
                ref_t = [x[0] for x in ref_data]

                ref_f0.insert(0, 0)
                ref_t.insert(0, ref_t[0] - 0.001)

                ref_f0.append(0)
                ref_t.append(ref_t[-1] + 0.001)

                ref_f0.insert(0, 0)
                ref_f0.append(0)

                ref_t.insert(0, 0)
                ref_t.append(len(y) / _SAMPLE_RATE)

                target_t = np.arange(0.032, max(ref_t), 0.01)

                f = interpolate.interp1d(ref_t, ref_f0, 'nearest')
                interp_f0 = f(target_t).astype(np.float32)

                _plot_audio(y, sr, interp_f0, target_t)

                if y.dtype != np.dtype(np.int16):
                    y = (y * np.iinfo(np.int16).max).astype(np.int16)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'data': _bytes_feature(y.tobytes()),
                    'data_sampling_rate': _int64_feature([sr]),
                    'data_num_channels': _int64_feature([1]),
                    'data_width': _int64_feature([len(y)]),
                    'pitch': _float_feature(interp_f0),
                    'pitch_confidence': _float_feature(np.ones(len(interp_f0))),
                }))

                out.write(example.SerializeToString())

    print('done')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _plot_audio(y, sr, f0, f0_t):
    x = [x * 1 / sr for x in range(len(y))]
    plt.plot(x, y, 'r')
    plt.ylim(-1, 1)
    plt.twinx()
    plt.scatter(x=f0_t, y=f0, s=6)
    plt.ylim(0, 350)
    plt.show()


main()

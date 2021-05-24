import sys
import librosa
import numpy as np
import pysptk
import matplotlib.pyplot as plt

from ..util import conversions


_SAMPLE_RATE = 16000
_HOP_SIZE = 512
_PITCH_TIME = 0.01
_PITCH_RATE = 1 // _PITCH_TIME


def get_file_path():
    if len(sys.argv) != 3:
        print('Please provide a wav file as the first and a reference pitch file " \
            + "as the second argument to this script.')
        sys.exit(-1)
    return sys.argv[1], sys.argv[2]


def pitch_estimation_ref_rapt(x, fs=_SAMPLE_RATE, hop_size=_HOP_SIZE):
    return pysptk.sptk.rapt(_adjust_value_space(x), fs, hop_size)


def pitch_estimation_ref_swipe(x, fs=_SAMPLE_RATE, hop_size=_HOP_SIZE):
    return pysptk.sptk.swipe(_adjust_value_space(x), fs, hop_size)


def _adjust_value_space(x):
    return x / np.max(np.abs(x)) * (2 ** 15 - 1)


def plot_pitch_estimates(audio, f0_ref, f0_ref_times, f0_rapt, f0_rapt_times, f0_swipe, f0_swipe_times):
    fig, ax = plt.subplots(sharey='all', sharex='all')
    # fig.tight_layout(h_pad=2, pad=3)
    # fig.suptitle('Speech spectrogram with ground truth')

    # plt.subplot(211)
    spec, freqs, times, colormap = plt.specgram(audio, Fs=_SAMPLE_RATE)  # 320 samples := 20ms offset (see readme)
    # plt.xlim(left=0.5, right=2)
    plt.ylim([0, 500])
    # plt.xlabel('Time [s]')
    # plt.ylabel('Frequency [Hz]')
    fig.colorbar(colormap).set_label('Intensity [dB]')

    # plt.subplot(212)
    plt.plot(f0_ref_times, f0_ref, 'red')
    plt.plot(f0_rapt_times, f0_rapt, 'blue')
    plt.plot(f0_swipe_times, f0_swipe, 'violet')
    plt.show()


def main():
    audio_path, pitch_path = get_file_path()

    f0 = np.genfromtxt(pitch_path, delimiter=' ')
    if False:
        f0 = conversions.convert_cent_to_hz(100 * f0, 10)  # convert cents (100*semitone) to hz

    f0_time = np.array([i / _PITCH_RATE for i in range(len(f0))])  # create time axis for f0

    f0_time = np.arange(0, len(f0)) * _PITCH_TIME

    if len(f0.shape) > 1:
        f0 = f0.T[0]  # for multi column ground truth files, assume f0 is in col1
    audio, sr_audio = librosa.load(path=audio_path, sr=_SAMPLE_RATE, mono=False)
    if len(audio.shape) == 2:
        audio = audio[1]  # for stereo audio, assume channel 1 is voice
    f0_rapt = pitch_estimation_ref_rapt(audio)
    f0_swipe = pitch_estimation_ref_swipe(audio.astype(np.float64))
    f0_estimated_time_base = np.arange(0, len(audio) // _HOP_SIZE + 1)
    f0_estimated_time = f0_estimated_time_base / _SAMPLE_RATE * _HOP_SIZE

    plot_pitch_estimates(audio, f0, f0_time, f0_rapt, f0_estimated_time, f0_swipe, f0_estimated_time)

    print(audio_path, pitch_path)


if __name__ == '__main__':
    main()
    sys.exit()

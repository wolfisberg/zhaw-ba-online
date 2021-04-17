import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import librosa
from librosa import display

SR_TARGET = 16000
data_path = os.path.join('..', 'data')
input_path = os.path.join(data_path, 'input')
output_path = os.path.join(data_path, 'output')
audio_file = os.path.join(input_path, 'abjones_1_01.wav')
pitch_file = os.path.join(input_path, 'abjones_1_01.pv')


def convert_cent_to_hz(c, fref=10.0):
    return fref * 2 ** (c / 1200.0)


f0 = np.genfromtxt(pitch_file, delimiter=' ')  # f0 ground truth in semitones
f0 = convert_cent_to_hz(100 * f0, 10)  # convert cents (100*semitone) to hz
f0_time = np.array([i / 50 for i in range(len(f0))])  # create time axis for f0
if len(f0.shape) > 1: f0 = f0[0]  # for multi column ground truth files, assume f0 is in col1
audio, sr_audio = librosa.load(path=audio_file, sr=SR_TARGET, mono=False)
if len(audio.shape) == 2: audio = audio[1]  # for stereo audio, assume channel 1 is voice

fig, ax = plt.subplots(2, 1)
fig.tight_layout(h_pad=2, pad=3)
fig.suptitle('Speech spectrogram with ground truth')

plt.subplot(211)
spec, freqs, times, colormap = plt.specgram(audio[320:], Fs=sr_audio)  # 320 samples := 20ms offset (see readme)
plt.xlim(left=0.5)
plt.ylim([0, 1500])
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
fig.colorbar(colormap).set_label('Intensity [dB]')

plt.subplot(212)
plt.plot(f0_time, f0, 'red')
spec, freqs, times, colormap = plt.specgram(audio[320:], Fs=sr_audio)  # 320 samples := 20ms offset (see readme)
plt.xlim(left=0.5)
plt.ylim([0, 1500])
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
fig.colorbar(colormap).set_label('Intensity [dB]')

plt.savefig(output_path, format='svg')
plt.show()

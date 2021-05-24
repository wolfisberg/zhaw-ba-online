import numpy as np
import librosa
import matplotlib.pyplot as plt
import sys


_SAMPLE_RATE = 16000


if __name__ == "__main__":
    wav_file = sys.argv[1]
    f0_file = sys.argv[2]

    y, sr = librosa.load(wav_file, sr=_SAMPLE_RATE, mono=True)
    wav_x = [x * 1 / _SAMPLE_RATE for x in range(len(y))]
    plt.plot(wav_x, y, 'r')
    plt.ylim(-1, 1)

    ref_data = np.genfromtxt(f0_file, delimiter=' ')
    f0_y = ref_data[:, 0]
    f0_x = [x * 0.01 + 0.32 for x in range(len(ref_data[:, 1]))]
    plt.twinx()
    plt.scatter(x=f0_x, y=f0_y, s=7)
    plt.ylim(0, 350)
    plt.show()
    exit(0)

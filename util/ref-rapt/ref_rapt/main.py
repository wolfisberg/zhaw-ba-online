import dataset
import matplotlib.pyplot as plt

import config
import estimations
import performance_metrics


"""
TODO
- Play with hop size
- Compare swipe
- Plot spectrogram
- Noisy vs Clean
"""


def main():
    dataset_test = dataset.get_test_data()

    for track in dataset_test:
        one = track
        one = one + (estimations.pitch_estimation_ref_rapt(one[0], config.FS),)

        mae = performance_metrics.mean_absolute_error(one)
        mse = performance_metrics.mean_square_error(one)
        std_dev = performance_metrics.standard_deviation_hz(one[1], one[2])

        plt.plot(one[1], 'g')
        plt.plot(one[2], 'r')
        plt.show()
        print('done.')

    print('done.')


if __name__ == '__main__':
    main()

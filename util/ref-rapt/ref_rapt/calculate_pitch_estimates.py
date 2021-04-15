import dataset
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime

import config
import estimations
import performance_metrics

"""
TODO
# - Play with hop size
# - Compare swipe
- Plot spectrogram
# - Noisy vs Clean
"""


def get_datetime_file_extension():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def get_rapt_estimates(audio, hop_size=config.FRAME_STEP):
    return estimations.pitch_estimation_ref_rapt(audio, config.FS, hop_size)


def get_swipe_estimation(audio, hop_size=config.FRAME_STEP):
    return estimations.pitch_estimation_ref_swipe(audio.astype(np.float64), config.FS, hop_size)


def get_pitch_estimates(dataset):
    estimates = []
    for item in dataset:
        estimate = {}
        estimate['audio'] = item[0]

        for i in range(len(config.FRAME_STEPS)):
            step_size = config.FRAME_STEPS[i]
            estimate[f'f0_{str(step_size)}'] = item[i + 1]
            estimate[f'f0_rapt_{str(step_size)}'] = get_rapt_estimates(item[0], step_size)
            estimate[f'f0_swipe_{str(step_size)}'] = get_swipe_estimation(item[0], step_size)

        estimates.append(estimate)

    return estimates


def main():
    dataset_test_clean = dataset.get_test_data(mix_with_noise=False)
    dataset_test_noisy = dataset.get_test_data(mix_with_noise=True)

    results = {}
    results['clean'] = get_pitch_estimates(dataset_test_clean)
    results['noisy'] = get_pitch_estimates(dataset_test_noisy)

    with open(f'results_{get_datetime_file_extension()}.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('done.')

    # mae = performance_metrics.mean_absolute_error(item)
    # mse = performance_metrics.mean_square_error(item)
    # std_dev = performance_metrics.standard_deviation_hz(item[1], item[2])


if __name__ == '__main__':
    main()

# plt.plot(item[1], 'g')
# plt.plot(item[2], 'r')
# plt.show()

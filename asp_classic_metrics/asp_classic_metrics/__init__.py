import numpy as np
import pickle
from rt_pie_lib import metrics


def get_metrics(filename):
    with open(f'{filename}_processed.pickle', 'rb') as handle:
        results = pickle.load(handle)

    frame_size = 256
    algo = 'rapt'
    hz_true = results[f'f0_{frame_size}']
    hz_pred = results[f'f0_{algo}_{frame_size}']

    print(f'######## metrics {frame_size}, {algo}')
    metrics.get_hz_metrics(hz_true, hz_pred, rpa_relative_tolerance=0.05, print_output=True)
    print('done')


def process_raw_pickle(filename):
    with open(f'{filename}.pickle', 'rb') as handle:
        results = pickle.load(handle)

    totals_noisy = {}

    for i in results['noisy']:
        for k in i:
            if k == 'audio':
                continue
            if k not in totals_noisy:
                totals_noisy[k] = np.array([])
            totals_noisy[k] = np.append(totals_noisy[k], i[k])

    with open(f'{filename}_processed.pickle', 'wb') as handle:
        pickle.dump(totals_noisy, handle)

    print('done')


def main():
    filename = 'results_20210608-235737'

    # process_raw_pickle(filename)
    get_metrics(filename)

    print('done')


if __name__ == '__main__':
    main()
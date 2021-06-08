import numpy as np
import pickle
from rt_pie_lib import metrics


def get_metrics(filename):
    with open(f'{filename}_processed.pickle', 'rb') as handle:
        results = pickle.load(handle)

    frame_sizes = [1024, 512, 256]
    algos = ['rapt', 'swipe']

    for algo in algos:
        for frame_size in frame_sizes:
            hz_true = results[f'f0_{frame_size}']
            hz_pred = results[f'f0_{algo}_{frame_size}']
            print(f'######## metrics {frame_size}, {algo}')
            metrics.get_hz_metrics(hz_true, hz_pred, rpa_relative_tolerance=0.05, print_output=True)
            print('')
            get_zpa(hz_true, hz_pred)
            print('\n\n\n')


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


def get_zpa(true_hz, predicted_hz, cutoff=35):
    tn, tp, fn, fp = 0, 0, 0, 0

    for i in range(len(true_hz)):
        if true_hz[i] <= 35 and predicted_hz[i] <= cutoff:
            tp += 1
        if true_hz[i] > 35 and predicted_hz[i] > cutoff:
            tn += 1
        if true_hz[i] <= 35 and predicted_hz[i] > cutoff:
            fn += 1
        if true_hz[i] > 35 and predicted_hz[i] <= cutoff:
            fp += 1

    if not all([tp, tn, fp, fn]):
        if tp == 0:
            tp = 1
        if tn == 0:
            tn = 1
        if fp == 0:
            fp = 1
        if fn == 0:
            fn = 1

    # try:
    sum = tp + fp + tn + fn
    percentage_zero_truth = (tp + fn) / sum * 100
    percentage_zero_predicted = (tp + fp) / sum * 100
    precision = tp / (tp + fp) * 100  # Anteil unserer 0 schätzungen die richtig sind
    recall = tp / (tp + fn) * 100  # Wieviele der tatsächlichen 0 schätzungen haben wir erwischt
    accuracy = (tp + tn) / sum * 100  # Anteil richtige predictions
    f1 = 2 * (precision * recall) / (precision + recall)

    tn_percentage = tn / sum * 100
    tp_percentage = tp / sum * 100
    fp_percentage = fp / sum * 100
    fn_percentage = fn / sum * 100

    print("ZERO PITCH ANALYSIS")
    print("Sample size (test data set): ", sum)
    print("0 - % in ground truth: ", "%.2f" % percentage_zero_truth)
    print("0 - % in predictions: ", "%.2f" % percentage_zero_predicted)
    print("Accuarcy: ", "%.2f" % accuracy)
    print("Precision: ", "%.2f" % precision)
    print("Recall: ", "%.2f" % recall)
    print("F1-Score", "%.2f" % f1)
    print("True Negatives: ", "%.2f" % tn_percentage)
    print("True Positives: ", "%.2f" % tp_percentage)
    print("False Positives: ", "%.2f" % fp_percentage)
    print("False Negatives: ", "%.2f" % fn_percentage)


def main():
    filename = 'results_20210608-235737'

    # process_raw_pickle(filename)
    get_metrics(filename)

    print('done')


if __name__ == '__main__':
    main()
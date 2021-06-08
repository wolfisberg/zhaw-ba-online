import numpy as np

from asp_classic import audio_util as au


def mean_absolute_error(pitch, estimate):
    return sum(abs(pitch - estimate)) / len(estimate)


def mean_square_error(pitch, estimate):
    return sum(np.square(pitch - estimate)) / len(estimate)


def standard_deviation_hz(true_hz, predicted_hz):
    diff = abs(predicted_hz - true_hz)
    avg = np.mean(diff)
    diff = np.square(diff - avg)
    sum = np.sum(diff)
    return np.sqrt((sum / (len(diff) - 1)))


# def raw_pitch_accuracy_cent(true_cents, predicted_cents, cent_tolerence=50):
#     counter_true = 0
#     counter_false = 0
#     for i in range(len(true_cents)):
#         if abs(predicted_cents[i] - true_cents[i]) <= cent_tolerence:
#             counter_true += 1
#         else:
#             counter_false += 1
#     return counter_true / (counter_true + counter_false) * 100


def __rpa_tolerance_function_relative(cent_true, cent_pred, tolerance):
    return abs(cent_true - cent_pred) <= (cent_true * tolerance)


def __raw_pitch_accuracy(cents_true, cents_pred, tolerance_function, tolerance):
    counter_true = 0
    counter_false = 0
    for i in range(len(cents_true)):
        if tolerance_function(cents_true[i], cents_pred[i], tolerance):
            counter_true += 1
        else:
            counter_false += 1
    if counter_true > 0:
        result = counter_true / (counter_true + counter_false) * 100
    else:
        result = 0
    return result


def raw_pitch_accuracy_cent(hz_true, hz_pred, relative_tolerance=0.05):
    return __raw_pitch_accuracy(hz_true, hz_pred, __rpa_tolerance_function_relative, relative_tolerance)


def calculate_performance_metrics(results):
    totals_clean = {}
    totals_noisy = {}

    for i in results['clean']:
        for k in i:
            if k == 'audio':
                continue
            if k not in totals_clean:
                totals_clean[k] = np.array([])
            totals_clean[k] = np.append(totals_clean[k], i[k])

    for i in results['noisy']:
        for k in i:
            if k == 'audio':
                continue
            if k not in totals_noisy:
                totals_noisy[k] = np.array([])
            totals_noisy[k] = np.append(totals_noisy[k], i[k])

    metrics = {}

    ref_64 = totals_clean['f0_64']
    rapt_64 = totals_clean['f0_rapt_64']
    swipe_64 = totals_clean['f0_swipe_64']
    metrics['clean_rapt_64_mae'] = mean_absolute_error(ref_64, rapt_64)
    metrics['clean_swipe_64_mae'] = mean_absolute_error(ref_64, swipe_64)
    # metrics['clean_rapt_64_mse'] = mean_square_error(ref_64, rapt_64)
    # metrics['clean_swipe_64_mse'] = mean_square_error(ref_64, swipe_64)
    metrics['clean_rapt_64_stddev'] = standard_deviation_hz(ref_64, rapt_64)
    metrics['clean_swipe_64_stddev'] = standard_deviation_hz(ref_64, swipe_64)
    metrics['clean_rapt_64_rpa'] = raw_pitch_accuracy_cent(ref_64, rapt_64)
    metrics['clean_swipe_64_rpa'] = raw_pitch_accuracy_cent(ref_64, swipe_64)

    ref_128 = totals_clean['f0_128']
    rapt_128 = totals_clean['f0_rapt_128']
    swipe_128 = totals_clean['f0_swipe_128']
    metrics['clean_rapt_128_mae'] = mean_absolute_error(ref_128, rapt_128)
    metrics['clean_swipe_128_mae'] = mean_absolute_error(ref_128, swipe_128)
    # metrics['clean_rapt_128_mse'] = mean_square_error(ref_128, rapt_128)
    # metrics['clean_swipe_128_mse'] = mean_square_error(ref_128, swipe_128)
    metrics['clean_rapt_128_stddev'] = standard_deviation_hz(ref_128, rapt_128)
    metrics['clean_swipe_128_stddev'] = standard_deviation_hz(ref_128, swipe_128)
    metrics['clean_rapt_128_rpa'] = raw_pitch_accuracy_cent(ref_128, rapt_128)
    metrics['clean_swipe_128_rpa'] = raw_pitch_accuracy_cent(ref_128, swipe_128)

    ref_256 = totals_clean['f0_256']
    rapt_256 = totals_clean['f0_rapt_256']
    swipe_256 = totals_clean['f0_swipe_256']
    metrics['clean_rapt_256_mae'] = mean_absolute_error(ref_256, rapt_256)
    metrics['clean_swipe_256_mae'] = mean_absolute_error(ref_256, swipe_256)
    # metrics['clean_rapt_256_mse'] = mean_square_error(ref_256, rapt_256)
    # metrics['clean_swipe_256_mse'] = mean_square_error(ref_256, swipe_256)
    metrics['clean_rapt_256_stddev'] = standard_deviation_hz(ref_256, rapt_256)
    metrics['clean_swipe_256_stddev'] = standard_deviation_hz(ref_256, swipe_256)
    metrics['clean_rapt_256_rpa'] = raw_pitch_accuracy_cent(ref_256, rapt_256)
    metrics['clean_swipe_256_rpa'] = raw_pitch_accuracy_cent(ref_256, swipe_256)

    ref_512 = totals_clean['f0_512']
    rapt_512 = totals_clean['f0_rapt_512']
    swipe_512 = totals_clean['f0_swipe_512']
    metrics['clean_rapt_512_mae'] = mean_absolute_error(ref_512, rapt_512)
    metrics['clean_swipe_512_mae'] = mean_absolute_error(ref_512, swipe_512)
    # metrics['clean_rapt_512_mse'] = mean_square_error(ref_512, rapt_512)
    # metrics['clean_swipe_512_mse'] = mean_square_error(ref_512, swipe_512)
    metrics['clean_rapt_512_stddev'] = standard_deviation_hz(ref_512, rapt_512)
    metrics['clean_swipe_512_stddev'] = standard_deviation_hz(ref_512, swipe_512)
    metrics['clean_rapt_512_rpa'] = raw_pitch_accuracy_cent(ref_512, rapt_512)
    metrics['clean_swipe_512_rpa'] = raw_pitch_accuracy_cent(ref_512, swipe_512)

    i = 5 + 5

    ref_1024 = totals_clean['f0_1024']
    rapt_1024 = totals_clean['f0_rapt_1024']
    swipe_1024 = totals_clean['f0_swipe_1024']
    metrics['clean_rapt_1024_mae'] = mean_absolute_error(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_mae'] = mean_absolute_error(ref_1024, swipe_1024)
    # metrics['clean_rapt_1024_mse'] = mean_square_error(ref_1024, rapt_1024)
    # metrics['clean_swipe_1024_mse'] = mean_square_error(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_stddev'] = standard_deviation_hz(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_stddev'] = standard_deviation_hz(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_rpa'] = raw_pitch_accuracy_cent(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_rpa'] = raw_pitch_accuracy_cent(ref_1024, swipe_1024)

    ref_64 = totals_clean['f0_64']
    rapt_64 = totals_clean['f0_rapt_64']
    swipe_64 = totals_clean['f0_swipe_64']
    metrics['clean_rapt_64_mae'] = mean_absolute_error(ref_64, rapt_64)
    metrics['clean_swipe_64_mae'] = mean_absolute_error(ref_64, swipe_64)
    # metrics['clean_rapt_64_mse'] = mean_square_error(ref_64, rapt_64)
    # metrics['clean_swipe_64_mse'] = mean_square_error(ref_64, swipe_64)
    metrics['clean_rapt_64_stddev'] = standard_deviation_hz(ref_64, rapt_64)
    metrics['clean_swipe_64_stddev'] = standard_deviation_hz(ref_64, swipe_64)
    metrics['clean_rapt_64_rpa'] = raw_pitch_accuracy_cent(ref_64, rapt_64)
    metrics['clean_swipe_64_rpa'] = raw_pitch_accuracy_cent(ref_64, swipe_64)

    ref_128 = totals_clean['f0_128']
    rapt_128 = totals_clean['f0_rapt_128']
    swipe_128 = totals_clean['f0_swipe_128']
    metrics['clean_rapt_128_mae'] = mean_absolute_error(ref_128, rapt_128)
    metrics['clean_swipe_128_mae'] = mean_absolute_error(ref_128, swipe_128)
    # metrics['clean_rapt_128_mse'] = mean_square_error(ref_128, rapt_128)
    # metrics['clean_swipe_128_mse'] = mean_square_error(ref_128, swipe_128)
    metrics['clean_rapt_128_stddev'] = standard_deviation_hz(ref_128, rapt_128)
    metrics['clean_swipe_128_stddev'] = standard_deviation_hz(ref_128, swipe_128)
    metrics['clean_rapt_128_rpa'] = raw_pitch_accuracy_cent(ref_128, rapt_128)
    metrics['clean_swipe_128_rpa'] = raw_pitch_accuracy_cent(ref_128, swipe_128)

    ref_256 = totals_clean['f0_256']
    rapt_256 = totals_clean['f0_rapt_256']
    swipe_256 = totals_clean['f0_swipe_256']
    metrics['clean_rapt_256_mae'] = mean_absolute_error(ref_256, rapt_256)
    metrics['clean_swipe_256_mae'] = mean_absolute_error(ref_256, swipe_256)
    # metrics['clean_rapt_256_mse'] = mean_square_error(ref_256, rapt_256)
    # metrics['clean_swipe_256_mse'] = mean_square_error(ref_256, swipe_256)
    metrics['clean_rapt_256_stddev'] = standard_deviation_hz(ref_256, rapt_256)
    metrics['clean_swipe_256_stddev'] = standard_deviation_hz(ref_256, swipe_256)
    metrics['clean_rapt_256_rpa'] = raw_pitch_accuracy_cent(ref_256, rapt_256)
    metrics['clean_swipe_256_rpa'] = raw_pitch_accuracy_cent(ref_256, swipe_256)

    ref_512 = totals_clean['f0_512']
    rapt_512 = totals_clean['f0_rapt_512']
    swipe_512 = totals_clean['f0_swipe_512']
    metrics['clean_rapt_512_mae'] = mean_absolute_error(ref_512, rapt_512)
    metrics['clean_swipe_512_mae'] = mean_absolute_error(ref_512, swipe_512)
    # metrics['clean_rapt_512_mse'] = mean_square_error(ref_512, rapt_512)
    # metrics['clean_swipe_512_mse'] = mean_square_error(ref_512, swipe_512)
    metrics['clean_rapt_512_stddev'] = standard_deviation_hz(ref_512, rapt_512)
    metrics['clean_swipe_512_stddev'] = standard_deviation_hz(ref_512, swipe_512)
    metrics['clean_rapt_512_rpa'] = raw_pitch_accuracy_cent(ref_512, rapt_512)
    metrics['clean_swipe_512_rpa'] = raw_pitch_accuracy_cent(ref_512, swipe_512)

    ref_1024 = totals_clean['f0_1024']
    rapt_1024 = totals_clean['f0_rapt_1024']
    swipe_1024 = totals_clean['f0_swipe_1024']
    metrics['clean_rapt_1024_mae'] = mean_absolute_error(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_mae'] = mean_absolute_error(ref_1024, swipe_1024)
    # metrics['clean_rapt_1024_mse'] = mean_square_error(ref_1024, rapt_1024)
    # metrics['clean_swipe_1024_mse'] = mean_square_error(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_stddev'] = standard_deviation_hz(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_stddev'] = standard_deviation_hz(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_rpa'] = raw_pitch_accuracy_cent(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_rpa'] = raw_pitch_accuracy_cent(ref_1024, swipe_1024)

    ref_64 = totals_noisy['f0_64']
    rapt_64 = totals_noisy['f0_rapt_64']
    swipe_64 = totals_noisy['f0_swipe_64']
    metrics['noisy_rapt_64_mae'] = mean_absolute_error(ref_64, rapt_64)
    metrics['noisy_swipe_64_mae'] = mean_absolute_error(ref_64, swipe_64)
    # metrics['noisy_rapt_64_mse'] = mean_square_error(ref_64, rapt_64)
    # metrics['noisy_swipe_64_mse'] = mean_square_error(ref_64, swipe_64)
    metrics['noisy_rapt_64_stddev'] = standard_deviation_hz(ref_64, rapt_64)
    metrics['noisy_swipe_64_stddev'] = standard_deviation_hz(ref_64, swipe_64)
    metrics['noisy_rapt_64_rpa'] = raw_pitch_accuracy_cent(ref_64, rapt_64)
    metrics['noisy_swipe_64_rpa'] = raw_pitch_accuracy_cent(ref_64, swipe_64)

    ref_128 = totals_noisy['f0_128']
    rapt_128 = totals_noisy['f0_rapt_128']
    swipe_128 = totals_noisy['f0_swipe_128']
    metrics['noisy_rapt_128_mae'] = mean_absolute_error(ref_128, rapt_128)
    metrics['noisy_swipe_128_mae'] = mean_absolute_error(ref_128, swipe_128)
    # metrics['noisy_rapt_128_mse'] = mean_square_error(ref_128, rapt_128)
    # metrics['noisy_swipe_128_mse'] = mean_square_error(ref_128, swipe_128)
    metrics['noisy_rapt_128_stddev'] = standard_deviation_hz(ref_128, rapt_128)
    metrics['noisy_swipe_128_stddev'] = standard_deviation_hz(ref_128, swipe_128)
    metrics['noisy_rapt_128_rpa'] = raw_pitch_accuracy_cent(ref_128, rapt_128)
    metrics['noisy_swipe_128_rpa'] = raw_pitch_accuracy_cent(ref_128, swipe_128)

    ref_256 = totals_noisy['f0_256']
    rapt_256 = totals_noisy['f0_rapt_256']
    swipe_256 = totals_noisy['f0_swipe_256']
    metrics['noisy_rapt_256_mae'] = mean_absolute_error(ref_256, rapt_256)
    metrics['noisy_swipe_256_mae'] = mean_absolute_error(ref_256, swipe_256)
    # metrics['noisy_rapt_256_mse'] = mean_square_error(ref_256, rapt_256)
    # metrics['noisy_swipe_256_mse'] = mean_square_error(ref_256, swipe_256)
    metrics['noisy_rapt_256_stddev'] = standard_deviation_hz(ref_256, rapt_256)
    metrics['noisy_swipe_256_stddev'] = standard_deviation_hz(ref_256, swipe_256)
    metrics['noisy_rapt_256_rpa'] = raw_pitch_accuracy_cent(ref_256, rapt_256)
    metrics['noisy_swipe_256_rpa'] = raw_pitch_accuracy_cent(ref_256, swipe_256)

    ref_512 = totals_noisy['f0_512']
    rapt_512 = totals_noisy['f0_rapt_512']
    swipe_512 = totals_noisy['f0_swipe_512']
    metrics['noisy_rapt_512_mae'] = mean_absolute_error(ref_512, rapt_512)
    metrics['noisy_swipe_512_mae'] = mean_absolute_error(ref_512, swipe_512)
    # metrics['noisy_rapt_512_mse'] = mean_square_error(ref_512, rapt_512)
    # metrics['noisy_swipe_512_mse'] = mean_square_error(ref_512, swipe_512)
    metrics['noisy_rapt_512_stddev'] = standard_deviation_hz(ref_512, rapt_512)
    metrics['noisy_swipe_512_stddev'] = standard_deviation_hz(ref_512, swipe_512)
    metrics['noisy_rapt_512_rpa'] = raw_pitch_accuracy_cent(ref_512, rapt_512)
    metrics['noisy_swipe_512_rpa'] = raw_pitch_accuracy_cent(ref_512, swipe_512)

    ref_1024 = totals_noisy['f0_1024']
    rapt_1024 = totals_noisy['f0_rapt_1024']
    swipe_1024 = totals_noisy['f0_swipe_1024']
    metrics['noisy_rapt_1024_mae'] = mean_absolute_error(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_mae'] = mean_absolute_error(ref_1024, swipe_1024)
    # metrics['noisy_rapt_1024_mse'] = mean_square_error(ref_1024, rapt_1024)
    # metrics['noisy_swipe_1024_mse'] = mean_square_error(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_stddev'] = standard_deviation_hz(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_stddev'] = standard_deviation_hz(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_rpa'] = raw_pitch_accuracy_cent(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_rpa'] = raw_pitch_accuracy_cent(ref_1024, swipe_1024)

    ref_64 = totals_noisy['f0_64']
    rapt_64 = totals_noisy['f0_rapt_64']
    swipe_64 = totals_noisy['f0_swipe_64']
    metrics['noisy_rapt_64_mae'] = mean_absolute_error(ref_64, rapt_64)
    metrics['noisy_swipe_64_mae'] = mean_absolute_error(ref_64, swipe_64)
    # metrics['noisy_rapt_64_mse'] = mean_square_error(ref_64, rapt_64)
    # metrics['noisy_swipe_64_mse'] = mean_square_error(ref_64, swipe_64)
    metrics['noisy_rapt_64_stddev'] = standard_deviation_hz(ref_64, rapt_64)
    metrics['noisy_swipe_64_stddev'] = standard_deviation_hz(ref_64, swipe_64)
    metrics['noisy_rapt_64_rpa'] = raw_pitch_accuracy_cent(ref_64, rapt_64)
    metrics['noisy_swipe_64_rpa'] = raw_pitch_accuracy_cent(ref_64, swipe_64)

    ref_128 = totals_noisy['f0_128']
    rapt_128 = totals_noisy['f0_rapt_128']
    swipe_128 = totals_noisy['f0_swipe_128']
    metrics['noisy_rapt_128_mae'] = mean_absolute_error(ref_128, rapt_128)
    metrics['noisy_swipe_128_mae'] = mean_absolute_error(ref_128, swipe_128)
    # metrics['noisy_rapt_128_mse'] = mean_square_error(ref_128, rapt_128)
    # metrics['noisy_swipe_128_mse'] = mean_square_error(ref_128, swipe_128)
    metrics['noisy_rapt_128_stddev'] = standard_deviation_hz(ref_128, rapt_128)
    metrics['noisy_swipe_128_stddev'] = standard_deviation_hz(ref_128, swipe_128)
    metrics['noisy_rapt_128_rpa'] = raw_pitch_accuracy_cent(ref_128, rapt_128)
    metrics['noisy_swipe_128_rpa'] = raw_pitch_accuracy_cent(ref_128, swipe_128)

    ref_256 = totals_noisy['f0_256']
    rapt_256 = totals_noisy['f0_rapt_256']
    swipe_256 = totals_noisy['f0_swipe_256']
    metrics['noisy_rapt_256_mae'] = mean_absolute_error(ref_256, rapt_256)
    metrics['noisy_swipe_256_mae'] = mean_absolute_error(ref_256, swipe_256)
    # metrics['noisy_rapt_256_mse'] = mean_square_error(ref_256, rapt_256)
    # metrics['noisy_swipe_256_mse'] = mean_square_error(ref_256, swipe_256)
    metrics['noisy_rapt_256_stddev'] = standard_deviation_hz(ref_256, rapt_256)
    metrics['noisy_swipe_256_stddev'] = standard_deviation_hz(ref_256, swipe_256)
    metrics['noisy_rapt_256_rpa'] = raw_pitch_accuracy_cent(ref_256, rapt_256)
    metrics['noisy_swipe_256_rpa'] = raw_pitch_accuracy_cent(ref_256, swipe_256)

    ref_512 = totals_noisy['f0_512']
    rapt_512 = totals_noisy['f0_rapt_512']
    swipe_512 = totals_noisy['f0_swipe_512']
    metrics['noisy_rapt_512_mae'] = mean_absolute_error(ref_512, rapt_512)
    metrics['noisy_swipe_512_mae'] = mean_absolute_error(ref_512, swipe_512)
    # metrics['noisy_rapt_512_mse'] = mean_square_error(ref_512, rapt_512)
    # metrics['noisy_swipe_512_mse'] = mean_square_error(ref_512, swipe_512)
    metrics['noisy_rapt_512_stddev'] = standard_deviation_hz(ref_512, rapt_512)
    metrics['noisy_swipe_512_stddev'] = standard_deviation_hz(ref_512, swipe_512)
    metrics['noisy_rapt_512_rpa'] = raw_pitch_accuracy_cent(ref_512, rapt_512)
    metrics['noisy_swipe_512_rpa'] = raw_pitch_accuracy_cent(ref_512, swipe_512)

    ref_1024 = totals_noisy['f0_1024']
    rapt_1024 = totals_noisy['f0_rapt_1024']
    swipe_1024 = totals_noisy['f0_swipe_1024']
    metrics['noisy_rapt_1024_mae'] = mean_absolute_error(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_mae'] = mean_absolute_error(ref_1024, swipe_1024)
    # metrics['noisy_rapt_1024_mse'] = mean_square_error(ref_1024, rapt_1024)
    # metrics['noisy_swipe_1024_mse'] = mean_square_error(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_stddev'] = standard_deviation_hz(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_stddev'] = standard_deviation_hz(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_rpa'] = raw_pitch_accuracy_cent(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_rpa'] = raw_pitch_accuracy_cent(ref_1024, swipe_1024)

    return metrics

import numpy as np

import audio_util as au


def _mean_absolute_error(pitch, estimate):
    return sum(abs(pitch - estimate)) / len(estimate)


def _mean_square_error(pitch, estimate):
    return sum(np.square(pitch - estimate)) / len(estimate)


def _standard_deviation_hz(true_hz, predicted_hz):
    diff = abs(predicted_hz - true_hz)
    avg = np.mean(diff)
    diff = np.square(diff - avg)
    sum = np.sum(diff)
    return np.sqrt((sum / (len(diff) - 1)))


def _raw_pitch_accuracy_cent(true_cents, predicted_cents, cent_tolerence=50):
    counter_true = 0
    counter_false = 0
    for i in range(len(true_cents)):
        if abs(predicted_cents[i] - true_cents[i]) <= cent_tolerence:
            counter_true += 1
        else:
            counter_false += 1
    return counter_true / (counter_true + counter_false) * 100


def calculate_performance_metrics(results):
    totals_clean = {}
    totals_noisy = {}

    for i in results['clean']:
        for k in i:
            if k == 'audio':
                continue
            if not k in totals_clean:
                totals_clean[k] = np.array([])
            totals_clean[k] = np.append(totals_clean[k], i[k])

    for i in results['noisy']:
        for k in i:
            if k == 'audio':
                continue
            if not k in totals_noisy:
                totals_noisy[k] = np.array([])
            totals_noisy[k] = np.append(totals_noisy[k], i[k])

    metrics = {}

    ref_64 = totals_clean['f0_64']
    rapt_64 = totals_clean['f0_rapt_64']
    swipe_64 = totals_clean['f0_swipe_64']
    metrics['clean_rapt_64_mae'] = _mean_absolute_error(ref_64, rapt_64)
    metrics['clean_swipe_64_mae'] = _mean_absolute_error(ref_64, swipe_64)
    metrics['clean_rapt_64_mse'] = _mean_square_error(ref_64, rapt_64)
    metrics['clean_swipe_64_mse'] = _mean_square_error(ref_64, swipe_64)
    metrics['clean_rapt_64_stddev'] = _standard_deviation_hz(ref_64, rapt_64)
    metrics['clean_swipe_64_stddev'] = _standard_deviation_hz(ref_64, swipe_64)
    metrics['clean_rapt_64_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_64),
                                                            au.convert_hz_to_cent(rapt_64))
    metrics['clean_swipe_64_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_64),
                                                            au.convert_hz_to_cent(swipe_64))

    ref_128 = totals_clean['f0_128']
    rapt_128 = totals_clean['f0_rapt_128']
    swipe_128 = totals_clean['f0_swipe_128']
    metrics['clean_rapt_128_mae'] = _mean_absolute_error(ref_128, rapt_128)
    metrics['clean_swipe_128_mae'] = _mean_absolute_error(ref_128, swipe_128)
    metrics['clean_rapt_128_mse'] = _mean_square_error(ref_128, rapt_128)
    metrics['clean_swipe_128_mse'] = _mean_square_error(ref_128, swipe_128)
    metrics['clean_rapt_128_stddev'] = _standard_deviation_hz(ref_128, rapt_128)
    metrics['clean_swipe_128_stddev'] = _standard_deviation_hz(ref_128, swipe_128)
    metrics['clean_rapt_128_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_128),
                                                            au.convert_hz_to_cent(rapt_128))
    metrics['clean_swipe_128_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_128),
                                                             au.convert_hz_to_cent(swipe_128))

    ref_256 = totals_clean['f0_256']
    rapt_256 = totals_clean['f0_rapt_256']
    swipe_256 = totals_clean['f0_swipe_256']
    metrics['clean_rapt_256_mae'] = _mean_absolute_error(ref_256, rapt_256)
    metrics['clean_swipe_256_mae'] = _mean_absolute_error(ref_256, swipe_256)
    metrics['clean_rapt_256_mse'] = _mean_square_error(ref_256, rapt_256)
    metrics['clean_swipe_256_mse'] = _mean_square_error(ref_256, swipe_256)
    metrics['clean_rapt_256_stddev'] = _standard_deviation_hz(ref_256, rapt_256)
    metrics['clean_swipe_256_stddev'] = _standard_deviation_hz(ref_256, swipe_256)
    metrics['clean_rapt_256_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_256),
                                                             au.convert_hz_to_cent(rapt_256))
    metrics['clean_swipe_256_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_256),
                                                              au.convert_hz_to_cent(swipe_256))

    ref_512 = totals_clean['f0_512']
    rapt_512 = totals_clean['f0_rapt_512']
    swipe_512 = totals_clean['f0_swipe_512']
    metrics['clean_rapt_512_mae'] = _mean_absolute_error(ref_512, rapt_512)
    metrics['clean_swipe_512_mae'] = _mean_absolute_error(ref_512, swipe_512)
    metrics['clean_rapt_512_mse'] = _mean_square_error(ref_512, rapt_512)
    metrics['clean_swipe_512_mse'] = _mean_square_error(ref_512, swipe_512)
    metrics['clean_rapt_512_stddev'] = _standard_deviation_hz(ref_512, rapt_512)
    metrics['clean_swipe_512_stddev'] = _standard_deviation_hz(ref_512, swipe_512)
    metrics['clean_rapt_512_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_512),
                                                             au.convert_hz_to_cent(rapt_512))
    metrics['clean_swipe_512_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_512),
                                                              au.convert_hz_to_cent(swipe_512))

    ref_1024 = totals_clean['f0_1024']
    rapt_1024 = totals_clean['f0_rapt_1024']
    swipe_1024 = totals_clean['f0_swipe_1024']
    metrics['clean_rapt_1024_mae'] = _mean_absolute_error(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_mae'] = _mean_absolute_error(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_mse'] = _mean_square_error(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_mse'] = _mean_square_error(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_stddev'] = _standard_deviation_hz(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_stddev'] = _standard_deviation_hz(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_1024),
                                                             au.convert_hz_to_cent(rapt_1024))
    metrics['clean_swipe_1024_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_1024),
                                                              au.convert_hz_to_cent(swipe_1024))

    ref_64 = totals_clean['f0_64']
    rapt_64 = totals_clean['f0_rapt_64']
    swipe_64 = totals_clean['f0_swipe_64']
    metrics['clean_rapt_64_mae'] = _mean_absolute_error(ref_64, rapt_64)
    metrics['clean_swipe_64_mae'] = _mean_absolute_error(ref_64, swipe_64)
    metrics['clean_rapt_64_mse'] = _mean_square_error(ref_64, rapt_64)
    metrics['clean_swipe_64_mse'] = _mean_square_error(ref_64, swipe_64)
    metrics['clean_rapt_64_stddev'] = _standard_deviation_hz(ref_64, rapt_64)
    metrics['clean_swipe_64_stddev'] = _standard_deviation_hz(ref_64, swipe_64)
    metrics['clean_rapt_64_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_64),
                                                            au.convert_hz_to_cent(rapt_64))
    metrics['clean_swipe_64_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_64),
                                                             au.convert_hz_to_cent(swipe_64))

    ref_128 = totals_clean['f0_128']
    rapt_128 = totals_clean['f0_rapt_128']
    swipe_128 = totals_clean['f0_swipe_128']
    metrics['clean_rapt_128_mae'] = _mean_absolute_error(ref_128, rapt_128)
    metrics['clean_swipe_128_mae'] = _mean_absolute_error(ref_128, swipe_128)
    metrics['clean_rapt_128_mse'] = _mean_square_error(ref_128, rapt_128)
    metrics['clean_swipe_128_mse'] = _mean_square_error(ref_128, swipe_128)
    metrics['clean_rapt_128_stddev'] = _standard_deviation_hz(ref_128, rapt_128)
    metrics['clean_swipe_128_stddev'] = _standard_deviation_hz(ref_128, swipe_128)
    metrics['clean_rapt_128_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_128),
                                                             au.convert_hz_to_cent(rapt_128))
    metrics['clean_swipe_128_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_128),
                                                              au.convert_hz_to_cent(swipe_128))

    ref_256 = totals_clean['f0_256']
    rapt_256 = totals_clean['f0_rapt_256']
    swipe_256 = totals_clean['f0_swipe_256']
    metrics['clean_rapt_256_mae'] = _mean_absolute_error(ref_256, rapt_256)
    metrics['clean_swipe_256_mae'] = _mean_absolute_error(ref_256, swipe_256)
    metrics['clean_rapt_256_mse'] = _mean_square_error(ref_256, rapt_256)
    metrics['clean_swipe_256_mse'] = _mean_square_error(ref_256, swipe_256)
    metrics['clean_rapt_256_stddev'] = _standard_deviation_hz(ref_256, rapt_256)
    metrics['clean_swipe_256_stddev'] = _standard_deviation_hz(ref_256, swipe_256)
    metrics['clean_rapt_256_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_256),
                                                             au.convert_hz_to_cent(rapt_256))
    metrics['clean_swipe_256_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_256),
                                                              au.convert_hz_to_cent(swipe_256))

    ref_512 = totals_clean['f0_512']
    rapt_512 = totals_clean['f0_rapt_512']
    swipe_512 = totals_clean['f0_swipe_512']
    metrics['clean_rapt_512_mae'] = _mean_absolute_error(ref_512, rapt_512)
    metrics['clean_swipe_512_mae'] = _mean_absolute_error(ref_512, swipe_512)
    metrics['clean_rapt_512_mse'] = _mean_square_error(ref_512, rapt_512)
    metrics['clean_swipe_512_mse'] = _mean_square_error(ref_512, swipe_512)
    metrics['clean_rapt_512_stddev'] = _standard_deviation_hz(ref_512, rapt_512)
    metrics['clean_swipe_512_stddev'] = _standard_deviation_hz(ref_512, swipe_512)
    metrics['clean_rapt_512_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_512),
                                                             au.convert_hz_to_cent(rapt_512))
    metrics['clean_swipe_512_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_512),
                                                              au.convert_hz_to_cent(swipe_512))

    ref_1024 = totals_clean['f0_1024']
    rapt_1024 = totals_clean['f0_rapt_1024']
    swipe_1024 = totals_clean['f0_swipe_1024']
    metrics['clean_rapt_1024_mae'] = _mean_absolute_error(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_mae'] = _mean_absolute_error(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_mse'] = _mean_square_error(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_mse'] = _mean_square_error(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_stddev'] = _standard_deviation_hz(ref_1024, rapt_1024)
    metrics['clean_swipe_1024_stddev'] = _standard_deviation_hz(ref_1024, swipe_1024)
    metrics['clean_rapt_1024_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_1024),
                                                              au.convert_hz_to_cent(rapt_1024))
    metrics['clean_swipe_1024_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_1024),
                                                               au.convert_hz_to_cent(swipe_1024))

    ref_64 = totals_noisy['f0_64']
    rapt_64 = totals_noisy['f0_rapt_64']
    swipe_64 = totals_noisy['f0_swipe_64']
    metrics['noisy_rapt_64_mae'] = _mean_absolute_error(ref_64, rapt_64)
    metrics['noisy_swipe_64_mae'] = _mean_absolute_error(ref_64, swipe_64)
    metrics['noisy_rapt_64_mse'] = _mean_square_error(ref_64, rapt_64)
    metrics['noisy_swipe_64_mse'] = _mean_square_error(ref_64, swipe_64)
    metrics['noisy_rapt_64_stddev'] = _standard_deviation_hz(ref_64, rapt_64)
    metrics['noisy_swipe_64_stddev'] = _standard_deviation_hz(ref_64, swipe_64)
    metrics['noisy_rapt_64_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_64),
                                                            au.convert_hz_to_cent(rapt_64))
    metrics['noisy_swipe_64_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_64),
                                                             au.convert_hz_to_cent(swipe_64))

    ref_128 = totals_noisy['f0_128']
    rapt_128 = totals_noisy['f0_rapt_128']
    swipe_128 = totals_noisy['f0_swipe_128']
    metrics['noisy_rapt_128_mae'] = _mean_absolute_error(ref_128, rapt_128)
    metrics['noisy_swipe_128_mae'] = _mean_absolute_error(ref_128, swipe_128)
    metrics['noisy_rapt_128_mse'] = _mean_square_error(ref_128, rapt_128)
    metrics['noisy_swipe_128_mse'] = _mean_square_error(ref_128, swipe_128)
    metrics['noisy_rapt_128_stddev'] = _standard_deviation_hz(ref_128, rapt_128)
    metrics['noisy_swipe_128_stddev'] = _standard_deviation_hz(ref_128, swipe_128)
    metrics['noisy_rapt_128_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_128),
                                                             au.convert_hz_to_cent(rapt_128))
    metrics['noisy_swipe_128_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_128),
                                                              au.convert_hz_to_cent(swipe_128))

    ref_256 = totals_noisy['f0_256']
    rapt_256 = totals_noisy['f0_rapt_256']
    swipe_256 = totals_noisy['f0_swipe_256']
    metrics['noisy_rapt_256_mae'] = _mean_absolute_error(ref_256, rapt_256)
    metrics['noisy_swipe_256_mae'] = _mean_absolute_error(ref_256, swipe_256)
    metrics['noisy_rapt_256_mse'] = _mean_square_error(ref_256, rapt_256)
    metrics['noisy_swipe_256_mse'] = _mean_square_error(ref_256, swipe_256)
    metrics['noisy_rapt_256_stddev'] = _standard_deviation_hz(ref_256, rapt_256)
    metrics['noisy_swipe_256_stddev'] = _standard_deviation_hz(ref_256, swipe_256)
    metrics['noisy_rapt_256_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_256),
                                                             au.convert_hz_to_cent(rapt_256))
    metrics['noisy_swipe_256_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_256),
                                                              au.convert_hz_to_cent(swipe_256))

    ref_512 = totals_noisy['f0_512']
    rapt_512 = totals_noisy['f0_rapt_512']
    swipe_512 = totals_noisy['f0_swipe_512']
    metrics['noisy_rapt_512_mae'] = _mean_absolute_error(ref_512, rapt_512)
    metrics['noisy_swipe_512_mae'] = _mean_absolute_error(ref_512, swipe_512)
    metrics['noisy_rapt_512_mse'] = _mean_square_error(ref_512, rapt_512)
    metrics['noisy_swipe_512_mse'] = _mean_square_error(ref_512, swipe_512)
    metrics['noisy_rapt_512_stddev'] = _standard_deviation_hz(ref_512, rapt_512)
    metrics['noisy_swipe_512_stddev'] = _standard_deviation_hz(ref_512, swipe_512)
    metrics['noisy_rapt_512_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_512),
                                                             au.convert_hz_to_cent(rapt_512))
    metrics['noisy_swipe_512_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_512),
                                                              au.convert_hz_to_cent(swipe_512))

    ref_1024 = totals_noisy['f0_1024']
    rapt_1024 = totals_noisy['f0_rapt_1024']
    swipe_1024 = totals_noisy['f0_swipe_1024']
    metrics['noisy_rapt_1024_mae'] = _mean_absolute_error(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_mae'] = _mean_absolute_error(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_mse'] = _mean_square_error(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_mse'] = _mean_square_error(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_stddev'] = _standard_deviation_hz(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_stddev'] = _standard_deviation_hz(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_1024),
                                                              au.convert_hz_to_cent(rapt_1024))
    metrics['noisy_swipe_1024_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_1024),
                                                               au.convert_hz_to_cent(swipe_1024))

    ref_64 = totals_noisy['f0_64']
    rapt_64 = totals_noisy['f0_rapt_64']
    swipe_64 = totals_noisy['f0_swipe_64']
    metrics['noisy_rapt_64_mae'] = _mean_absolute_error(ref_64, rapt_64)
    metrics['noisy_swipe_64_mae'] = _mean_absolute_error(ref_64, swipe_64)
    metrics['noisy_rapt_64_mse'] = _mean_square_error(ref_64, rapt_64)
    metrics['noisy_swipe_64_mse'] = _mean_square_error(ref_64, swipe_64)
    metrics['noisy_rapt_64_stddev'] = _standard_deviation_hz(ref_64, rapt_64)
    metrics['noisy_swipe_64_stddev'] = _standard_deviation_hz(ref_64, swipe_64)
    metrics['noisy_rapt_64_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_64),
                                                            au.convert_hz_to_cent(rapt_64))
    metrics['noisy_swipe_64_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_64),
                                                             au.convert_hz_to_cent(swipe_64))

    ref_128 = totals_noisy['f0_128']
    rapt_128 = totals_noisy['f0_rapt_128']
    swipe_128 = totals_noisy['f0_swipe_128']
    metrics['noisy_rapt_128_mae'] = _mean_absolute_error(ref_128, rapt_128)
    metrics['noisy_swipe_128_mae'] = _mean_absolute_error(ref_128, swipe_128)
    metrics['noisy_rapt_128_mse'] = _mean_square_error(ref_128, rapt_128)
    metrics['noisy_swipe_128_mse'] = _mean_square_error(ref_128, swipe_128)
    metrics['noisy_rapt_128_stddev'] = _standard_deviation_hz(ref_128, rapt_128)
    metrics['noisy_swipe_128_stddev'] = _standard_deviation_hz(ref_128, swipe_128)
    metrics['noisy_rapt_128_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_128),
                                                             au.convert_hz_to_cent(rapt_128))
    metrics['noisy_swipe_128_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_128),
                                                              au.convert_hz_to_cent(swipe_128))

    ref_256 = totals_noisy['f0_256']
    rapt_256 = totals_noisy['f0_rapt_256']
    swipe_256 = totals_noisy['f0_swipe_256']
    metrics['noisy_rapt_256_mae'] = _mean_absolute_error(ref_256, rapt_256)
    metrics['noisy_swipe_256_mae'] = _mean_absolute_error(ref_256, swipe_256)
    metrics['noisy_rapt_256_mse'] = _mean_square_error(ref_256, rapt_256)
    metrics['noisy_swipe_256_mse'] = _mean_square_error(ref_256, swipe_256)
    metrics['noisy_rapt_256_stddev'] = _standard_deviation_hz(ref_256, rapt_256)
    metrics['noisy_swipe_256_stddev'] = _standard_deviation_hz(ref_256, swipe_256)
    metrics['noisy_rapt_256_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_256),
                                                             au.convert_hz_to_cent(rapt_256))
    metrics['noisy_swipe_256_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_256),
                                                              au.convert_hz_to_cent(swipe_256))

    ref_512 = totals_noisy['f0_512']
    rapt_512 = totals_noisy['f0_rapt_512']
    swipe_512 = totals_noisy['f0_swipe_512']
    metrics['noisy_rapt_512_mae'] = _mean_absolute_error(ref_512, rapt_512)
    metrics['noisy_swipe_512_mae'] = _mean_absolute_error(ref_512, swipe_512)
    metrics['noisy_rapt_512_mse'] = _mean_square_error(ref_512, rapt_512)
    metrics['noisy_swipe_512_mse'] = _mean_square_error(ref_512, swipe_512)
    metrics['noisy_rapt_512_stddev'] = _standard_deviation_hz(ref_512, rapt_512)
    metrics['noisy_swipe_512_stddev'] = _standard_deviation_hz(ref_512, swipe_512)
    metrics['noisy_rapt_512_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_512),
                                                             au.convert_hz_to_cent(rapt_512))
    metrics['noisy_swipe_512_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_512),
                                                              au.convert_hz_to_cent(swipe_512))

    ref_1024 = totals_noisy['f0_1024']
    rapt_1024 = totals_noisy['f0_rapt_1024']
    swipe_1024 = totals_noisy['f0_swipe_1024']
    metrics['noisy_rapt_1024_mae'] = _mean_absolute_error(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_mae'] = _mean_absolute_error(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_mse'] = _mean_square_error(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_mse'] = _mean_square_error(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_stddev'] = _standard_deviation_hz(ref_1024, rapt_1024)
    metrics['noisy_swipe_1024_stddev'] = _standard_deviation_hz(ref_1024, swipe_1024)
    metrics['noisy_rapt_1024_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_1024),
                                                              au.convert_hz_to_cent(rapt_1024))
    metrics['noisy_swipe_1024_rpa'] = _raw_pitch_accuracy_cent(au.convert_hz_to_cent(ref_1024),
                                                               au.convert_hz_to_cent(swipe_1024))

    return metrics

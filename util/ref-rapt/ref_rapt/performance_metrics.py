import numpy as np


def mean_absolute_error(pitch):
    return sum(abs(pitch[1] - pitch[2])) / len(pitch[1])


def mean_square_error(pitch):
    return sum(np.square(pitch[2] - pitch[1])) / len(pitch[1])


def standard_deviation_hz(true_hz, predicted_hz):
    diff = abs(predicted_hz - true_hz)
    avg = np.mean(diff)
    diff = np.square(diff - avg)
    sum = np.sum(diff)
    return np.sqrt((sum / (len(diff) - 1)))

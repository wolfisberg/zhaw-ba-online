import pysptk
import numpy as np

from asp_classic import config


def pitch_estimation_ref_rapt(x, fs, hop_size=config.FRAME_STEP):
    return pysptk.sptk.rapt(_adjust_value_space(x), fs, hop_size)


def pitch_estimation_ref_swipe(x, fs, hop_size=config.FRAME_STEP):
    return pysptk.sptk.swipe(_adjust_value_space(x), fs, hop_size)


def _adjust_value_space(x):
    return x / np.max(np.abs(x)) * (2 ** 15 - 1)

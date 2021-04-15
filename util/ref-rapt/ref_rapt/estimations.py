import pysptk
import numpy as np

import config


def pitch_estimation_ref_rapt(x, fs, hop_size=config.FRAME_STEP):
    return pysptk.sptk.rapt(x / np.max(np.abs(x)) * (2 ** 15 - 1), fs, hop_size)

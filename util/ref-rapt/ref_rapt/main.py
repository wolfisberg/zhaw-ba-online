import numpy as np
import dataset
import matplotlib.pyplot as plt

import config
import util


dataset_test = dataset.get_test_data()

one = dataset_test[0]
one = one + (util.pitch_estimation_ref_rapt(one[0], config.FS),)

mae = sum(abs(one[1] - one[2])) / len(one[1])
mse = sum(np.square(one[2] - one[1])) / len(one[1])

plt.plot(one[1], 'g')
plt.plot(one[2], 'r')
plt.show()

print('done.')

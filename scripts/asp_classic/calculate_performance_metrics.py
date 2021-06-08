import pickle
import numpy as np

from asp_classic import performance_metrics as pm

filename = 'results_20210608-231135'

with open(f'{filename}.pickle', 'rb') as handle:
    results = pickle.load(handle)

totals_clean = {}
totals_noisy = {}

# for i in results['clean']:
#     for k in i:
#         if k == 'audio':
#             continue
#         if k not in totals_clean:
#             totals_clean[k] = np.array([])
#         totals_clean[k] = np.append(totals_clean[k], i[k])

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

# metrics = pm.calculate_performance_metrics(results)

# [print(f'{k.ljust(25, "_")}{round(metrics[k], 2)}') for k in metrics
#     if k.count('noisy') > 0
#     and k.count('swipe') > 0
#     and k.count('512') > 0
#  ]
#
# print('')
#
# [print(f'{k.ljust(25, "_")}{round(metrics[k], 2)}') for k in metrics
#     if k.count('noisy') > 0
#     and k.count('rapt') > 0
#     and k.count('256') > 0
#  ]
#
# print('')
#
# [print(f'{k.ljust(25, "_")}{round(metrics[k], 2)}') for k in metrics
#     if k.count('clean') > 0
#     and k.count('swipe') > 0
#     and k.count('512') > 0
#  ]
#
# print('')
#
# [print(f'{k.ljust(25, "_")}{round(metrics[k], 2)}') for k in metrics
#     if k.count('clean') > 0
#     and k.count('rapt') > 0
#     and k.count('256') > 0
#  ]
#
# print('\ndone.')

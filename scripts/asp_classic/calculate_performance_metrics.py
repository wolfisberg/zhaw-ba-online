import pickle

from asp_classic import performance_metrics as pm

with open('results_20210415-193302.pickle', 'rb') as handle:
    results = pickle.load(handle)

metrics = pm.calculate_performance_metrics(results)

[print(f'{k.ljust(25, "_")}{round(metrics[k], 2)}') for k in metrics
    if k.count('noisy') > 0
    and k.count('swipe') > 0
    and k.count('512') > 0
 ]

print('')

[print(f'{k.ljust(25, "_")}{round(metrics[k], 2)}') for k in metrics
    if k.count('noisy') > 0
    and k.count('rapt') > 0
    and k.count('256') > 0
 ]

print('')

[print(f'{k.ljust(25, "_")}{round(metrics[k], 2)}') for k in metrics
    if k.count('clean') > 0
    and k.count('swipe') > 0
    and k.count('512') > 0
 ]

print('')

[print(f'{k.ljust(25, "_")}{round(metrics[k], 2)}') for k in metrics
    if k.count('clean') > 0
    and k.count('rapt') > 0
    and k.count('256') > 0
 ]

print('\ndone.')

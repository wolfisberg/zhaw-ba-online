import pickle

import performance_metrics as pm

# with open('results_20210415-191617.pickle', 'rb') as handle:
with open('results_20210415-181057.pickle', 'rb') as handle:
    results = pickle.load(handle)

metrics = pm.calculate_performance_metrics(results)

[print(f'{k.ljust(25, "-")}{round(metrics[k], 2)}') for k in metrics
 if k.count('noisy') > 0
 and k.count('swipe') > 0
 and k.count('512') > 0
 ]

print('')

[print(f'{k.ljust(25, "-")}{round(metrics[k], 2)}') for k in metrics
 if k.count('noisy') > 0
 and k.count('rapt') > 0
 and k.count('256') > 0
 ]

print('\ndone.')


# noisy - swipe - 512

# noisy rapt - 256
import pickle
from rt_pie_lib import metrics

filename = 'results_20210608-231135'


with open(f'{filename}_processed.pickle', 'rb') as handle:
    results = pickle.load(handle)

print('done')

import pickle


# with open(f'results_{get_datetime_file_extension()}.pickle', 'wb') as handle:
#     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('results_20210415-181057.pickle', 'rb') as handle:
    results = pickle.load(handle)

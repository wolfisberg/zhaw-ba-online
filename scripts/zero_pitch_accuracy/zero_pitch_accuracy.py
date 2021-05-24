import numpy as np


f0_truth = np.genfromtxt('zero_pitch_accuracy/test_groundtruth.f0')
f0_pred = np.genfromtxt('zero_pitch_accuracy/test_predictions.f0')
zip = zip(f0_truth, f0_pred)

# tn = len([x for x in zip if x[0] == 0 and x[1] == 0])
# tp = len([x for x in zip if x[0] > 0 and x[1] > 0])
# fp = len([x for x in zip if x[0] == 0 and x[1] != 0])
# fn = len([x for x in zip if x[0] != 0 and x[1] == 0])

tn = 0
tp = 0
fp = 0
fn = 0

for i in range(len(f0_truth)):
    if f0_truth[i] == 0 and f0_pred[i] == 0:
        tp += 1
        continue
    if f0_truth[i] > 0 and f0_pred[i] > 0:
        tn += 1
        continue
    if f0_truth[i] == 0 and f0_pred[i] > 0:
        fn += 1
        continue
    if f0_truth[i] > 0 and f0_pred[i] == 0:
        fp += 1
        continue
    print('error')


sum = tp + fp + tn + fn
percentage_zero_truth = (tp + fn) / sum
percentage_zero_predicted = (tp + fp) / sum
precision = tp / (tp + fp)  # Anteil unserer 0 schätzungen die richtig sind
recall = tp / (tp + fn)  # Wieviele der tatsächlichen 0 schätzungen haben wir erwischt
accuracy = (tp + tn) / sum  # Anteil richtige predictions
f1 = 2 * (precision * recall) / (precision + recall)

tn_percentage = tn / sum
tp_percentage = tp / sum
fp_percentage = fp / sum
fn_percentage = fn / sum

print('done')

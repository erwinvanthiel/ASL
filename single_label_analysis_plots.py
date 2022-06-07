import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
import numpy.ma as ma
from scipy.stats import spearmanr
mpl.style.use('classic')

model_type = 'asl'
dataset_type = 'NUS_WIDE'
num_classes = 81

flipped_labels = np.load('experiment_results/per-label-flips-{0}-{1}.npy'.format(model_type, dataset_type))
confidences = np.load('experiment_results/per-label-confidences-{0}-{1}.npy'.format(model_type, dataset_type))
frequencies = np.load('experiment_results/dataset-distribution-{0}.npy'.format(dataset_type))

ids = np.array([x for x in range(num_classes)])

flips = np.sum(flipped_labels, axis=1)
flip_stds = np.std(flipped_labels, axis=1)

confidence_means = np.mean(confidences, axis=1)
confidence_stds = np.std(confidences, axis=1)

normalised_confidences = confidence_means - np.min(confidence_means)
normalised_confidences = normalised_confidences / np.max(normalised_confidences)

normalised_flips = flips - np.min(flips)
normalised_flips = normalised_flips / np.max(normalised_flips)

normalised_frequencies = frequencies - np.min(frequencies)
normalised_frequencies = normalised_frequencies / np.max(normalised_frequencies)

# plt.bar(ids, normalised_frequencies)
# plt.show()
plt.plot(ids, normalised_flips, color='red')
plt.show()
plt.plot(ids, confidence_means, color='green')
plt.show()


flip_ranking = np.argsort(-1*normalised_flips)
confidence_ranking = np.argsort(-1*normalised_confidences)
frequency_ranking = np.argsort(-1*normalised_frequencies)

def calculate_correlation(sorted1, sorted2, num_classes):

	ranking1 = np.arange(num_classes)
	ranking2 = np.zeros(num_classes)

	for i in range(num_classes):
		item_in_1 = sorted1[i]
		ranking_of_i = np.where(sorted2 == item_in_1)[0]
		ranking2[i] = ranking_of_i

	return spearmanr(ranking1, ranking2)

print('R x C', calculate_correlation(flip_ranking, confidence_ranking, num_classes))
print('F x C', calculate_correlation(frequency_ranking, confidence_ranking, num_classes))
print('F x R', calculate_correlation(frequency_ranking, flip_ranking, num_classes))
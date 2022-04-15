import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

model_type = 'asl'
dataset_type = 'MSCOCO_2014'

flipped_labels = np.load('experiment_results/maxdist_singlelabel_flips-{0}-{1}.npy'.format(model_type, dataset_type))
confidences = np.load('experiment_results/maxdist_singlelabel_confidences-{0}-{1}.npy'.format(model_type, dataset_type))

ids = [x for x in range(80)]

flips = np.sum(flipped_labels, axis=1)
flip_stds = np.std(flipped_labels, axis=1)

confidence_means = np.mean(confidences, axis=1)
confidence_stds = np.std(confidences, axis=1)

plt.bar(ids, flips, yerr=None)
plt.show()
plt.bar(ids, confidence_means, yerr=None)
plt.show()
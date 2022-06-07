"""
=============================
Grouped bar chart with labels
=============================

This example shows a how to create a grouped bar chart and how to annotate
bars with labels.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn import metrics
mpl.style.use('classic')
model = 'q2l'
dataset = 'NUS_WIDE'

flipped_labels = np.load('experiment_results/explicit-flips-{0}-{1}.npy'.format(model, dataset))
epsilons = np.load('experiment_results/{0}-{1}-profile-epsilons.npy'.format(model, dataset))
max_eps = np.max(epsilons)
min_eps = np.min(max_eps) / 10
EPSILON_VALUES = [0.5*min_eps, min_eps, 2*min_eps, 4*min_eps, 6*min_eps, 8*min_eps, 10*min_eps]
print(EPSILON_VALUES)



# plt.hist(flipped_labels[0,0])
# plt.show()


means_0 = np.mean(flipped_labels,axis=2)[0]
means_1 = np.mean(flipped_labels,axis=2)[1]
means_2 = np.mean(flipped_labels,axis=2)[2]
means_3 = np.mean(flipped_labels,axis=2)[3]

std_0 = np.std(flipped_labels,axis=2)[0]
std_1 = np.std(flipped_labels,axis=2)[1]
std_2 = np.std(flipped_labels,axis=2)[2]
std_3 = np.std(flipped_labels,axis=2)[3]


print(np.column_stack((means_0,std_0)))
print("##############################################")
print(np.column_stack((means_1,std_1)))
print("##############################################")
print(np.column_stack((means_2,std_2)))
print("##############################################")
print(np.column_stack((means_3,std_3)))

# print(metrics.auc(EPSILON_VALUES, means_0))
# print(metrics.auc(EPSILON_VALUES, means_1))
# print(metrics.auc(EPSILON_VALUES, means_2))
# print(metrics.auc(EPSILON_VALUES, means_3))

print(np.sum(means_0))
print(np.sum(means_1))
print(np.sum(means_2))
print(np.sum(means_3))


domain_sums_asl_coco = [366.41, 362.02, 373.01, 366.96, 383.53, 371.92, 371.25]
domain_sums_q2l_coco = [301.16, 291.36, 306.27, 296.69, 306.35, 305.76, 305.73]
domain_sums_asl_nuswide = [357.76, 349.56, 364.75, 361.66, 365.49, 365.90, 365.29]
domain_sums_q2l_nuswide = [196.60, 236.03, 229.7, 192.10, 214.90, 203.69, 202.26]

labels = ['1', '2', '3', '4', '5', '6', '7']
plt.bar(labels, domain_sums_q2l_nuswide, color=('white','yellow','green', 'blue','purple','red','black'))
plt.ylim(0.95 * np.min(domain_sums_q2l_nuswide), 1.05 * np.max(domain_sums_q2l_nuswide))
plt.show()
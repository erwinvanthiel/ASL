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
model = 'asl'
dataset = 'VOC2007'

flipped_labels = np.load('../experiment_results/flips-{0}-{1}.npy'.format(model, dataset))
epsilons = np.load('../experiment_results/{0}-{1}-profile-epsilons.npy'.format(model, dataset))
max_eps = np.max(epsilons)
min_eps = np.min(max_eps) / 10
EPSILON_VALUES = [0.5*min_eps, min_eps, 2*min_eps, 4*min_eps, 6*min_eps, 8*min_eps, 10*min_eps]
# print(EPSILON_VALUES)


results = []
sums = []
for i in range(7):
	sums.append(np.sum(np.mean(flipped_labels, axis=2)[i]))
	stack = np.column_stack((np.mean(flipped_labels, axis=2)[i],np.std(flipped_labels, axis=2)[i]))
	print("########################")
	for j in range(7):
		print(round(stack[j][0],2), "$\\pm$", round(stack[j][1],2))

print(sums)

# domain_sums_asl_coco = [366.41, 362.02, 373.01, 366.96, 383.53, 371.92, 371.25] #370.56
# domain_sums_q2l_coco = [301.16, 291.36, 306.27, 296.69, 306.35, 305.76, 305.73] #305.31
# domain_sums_asl_nuswide = [357.76, 349.56, 364.75, 361.66, 365.49, 365.90, 365.29] #318.32
# domain_sums_q2l_nuswide = [196.60, 236.03, 229.7, 192.10, 214.90, 203.69, 202.26] #227.25

# labels = ['1', '2', '3', '4', '5', '6', '7']
# plt.bar(labels, domain_sums_q2l_nuswide, color=('white','yellow','green', 'blue','purple','red','black'))
# plt.ylim(0.95 * np.min(domain_sums_q2l_nuswide), 1.05 * np.max(domain_sums_q2l_nuswide))
# plt.show()
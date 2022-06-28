import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

model_type = 'q2l'
dataset_type = 'MSCOCO_2014'

matrix = np.load('experiment_results/tree-depth-x-branches-{0}-{1}.npy'.format(model_type, dataset_type))
means = np.mean(matrix[0], axis=2)
numbers_of_leaves = np.zeros(means.shape)

for i in range(means.shape[0]):
	for j in range(means.shape[1]):
		numbers_of_leaves[i,j] = (j + 1) ** (i + 1)

sns.heatmap(means, annot=numbers_of_leaves)
plt.show()
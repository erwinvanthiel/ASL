import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

matrix = np.load('experiment_results/tree-depth-x-branches.npy')
means = matrix[0,0,:,:,0]
stds = matrix[0,0,:,:,1]

print(means)
print(stds)
sns.heatmap(means)
plt.show()
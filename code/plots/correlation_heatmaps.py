import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
import seaborn as sns
mpl.style.use('classic')


dataset_type = 'NUS_WIDE'
model_type = 'asl'

correlations = np.load('experiment_results/flipup-correlations-cd-{0}-{1}.npy'.format(dataset_type, model_type))
sns.heatmap(correlations)
plt.show()


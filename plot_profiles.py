import os
import torch
import numpy.polynomial.polynomial as poly
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib as mpl
mpl.style.use('classic')

model_type = 'q2l'
dataset_type = 'MSCOCO_2014'

coefs = np.load('experiment_results/{0}-{1}-profile.npy'.format(model_type, dataset_type))
flips = np.load('experiment_results/{0}-{1}-profile-flips.npy'.format(model_type, dataset_type))
epsilons = np.load('experiment_results/{0}-{1}-profile-epsilons.npy'.format(model_type, dataset_type))
# flips = [0, 9.72, 18.11, 25.01, 29.99, 33.26, 36.68, 38.48, 40.22, 41.56, 42.39, 42.51, 43.19]
# epsilons = [0.0, 0.01953125, 0.0390625, 0.05859375, 0.078125, 0.09765625, 0.1171875, 0.13671875, 0.15625, 0.17578125, 0.1953125, 0.21484375, 0.234375]

coefs = poly.polyfit(epsilons, flips, 4)


xspace = np.linspace(0, epsilons[len(epsilons)-1])
poly = poly.polyval(xspace, coefs)
plt.scatter(epsilons, flips, label='data', color='r')
plt.plot(xspace, poly, label='fitted polynomial', color='green')
plt.ylim(0,90)
plt.xlim(0,np.max(epsilons))
plt.legend()
plt.show()

# np.save('experiment_results/{0}-{1}-profile-flips'.format(model_type, dataset_type), flips)
# np.save('experiment_results/{0}-{1}-profile'.format(model_type, dataset_type), coefs)
# np.save('experiment_results/{0}-{1}-profile-epsilons'.format(model_type, dataset_type), epsilons)

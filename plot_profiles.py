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
dataset_type = 'NUS_WIDE'

coefs = np.load('experiment_results/{0}-{1}-profile.npy'.format(model_type, dataset_type))
flips = np.load('experiment_results/{0}-{1}-profile-flips.npy'.format(model_type, dataset_type))
epsilons = np.load('experiment_results/{0}-{1}-profile-epsilons.npy'.format(model_type, dataset_type))

print(epsilons)
print(flips)
coefs = poly.polyfit(epsilons, flips, 4)
# np.save('experiment_results/{0}-{1}-profile'.format(model_type, dataset_type), coefs)


xspace = np.linspace(0, epsilons[len(epsilons)-1])
poly = poly.polyval(xspace, coefs)
plt.plot(epsilons, flips)
plt.plot(xspace, poly)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

# def least_squares_solution(points):

# 	# squares = sum for each point (ax - y)^2 
# 	# squares' = 0 so sum for each point 2(ax - y) * x = 2axx - 2xy = 0
# 	# a = sum xy / xx
# 	a0 = 0
# 	a1 = 0
# 	for p in points:
# 		y = p[0]
# 		x = p[1]
# 		a0 = a0 + (x*y)
# 		a1 = a1 + (x*x) 
# 	return a0/a1

model_type = 'q2l'
dataset_type = 'MSCOCO_2014'
num_classes = 80

amount_of_targets = [5,10,15,20,25,30,35,40,50,60,70,80]
flipped_labels = np.load('experiment_results/l2-targets-vs-flips-{0}-{1}.npy'.format(model_type, dataset_type))
flips_eps1 = np.mean(flipped_labels[0][0], axis=1)
flips_eps2 = np.mean(flipped_labels[0][1], axis=1)
flips_eps3 = np.mean(flipped_labels[0][2], axis=1)






plt.plot(amount_of_targets, flips_eps1)
plt.plot(amount_of_targets, flips_eps2)
plt.plot(amount_of_targets, flips_eps3)
plt.ylim(0, 80)
# plt.fill_between(amount_of_targets, flips-flip_stds, flips+flip_stds, alpha=0.5)
plt.legend()
# plt.ylim(0,30)
plt.show()


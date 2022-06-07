import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

def least_squares_solution(points):

	# squares = sum for each point (ax - y)^2 
	# squares' = 0 so sum for each point 2(ax - y) * x = 2axx - 2xy = 0
	# a = sum xy / xx
	a0 = 0
	a1 = 0
	for p in points:
		y = p[0]
		x = p[1]
		a0 = a0 + (x*y)
		a1 = a1 + (x*x) 
	return a0/a1

model_type = 'asl'
dataset_type = 'NUS_WIDE'
num_classes = 80
# amount_of_targets = [i for i in range(0,num_classes+1,10)]
amount_of_targets = [0,5,10,15,20,25,30,35,40,50,60,70,80]
flipped_labels = np.load('experiment_results/targets-vs-flips-{0}-{1}.npy'.format(model_type, dataset_type))

flips_eps1 = np.mean(flipped_labels[0][0], axis=1)
flips_eps2 = np.mean(flipped_labels[0][1], axis=1)
flips_eps3 = np.mean(flipped_labels[0][2], axis=1)
# flips_eps4 = np.mean(flipped_labels[0][3], axis=1)

# flips_eps1_r = np.mean(flipped_labels[1][0], axis=1)
# flips_eps2_r = np.mean(flipped_labels[1][1], axis=1)
# flips_eps3_r = np.mean(flipped_labels[1][2], axis=1)
# flips_eps4_r = np.mean(flipped_labels[1][3], axis=1)



# tops = []
# tops.append((np.max(flips_eps1), amount_of_targets[np.argmax(flips_eps1)]))
# tops.append((np.max(flips_eps2), amount_of_targets[np.argmax(flips_eps2)]))
# tops.append((np.max(flips_eps3), amount_of_targets[np.argmax(flips_eps3)]))


xspace = np.linspace(0,80)
# a = least_squares_solution(tops)
a = 1 / 1.66
line = a * xspace
# print(a)

# for t in tops:
	# plt.scatter(t[1], t[0])
plt.plot(amount_of_targets, flips_eps1, label='\u03B5 = 0.007')
plt.plot(xspace, line)
plt.plot(amount_of_targets, flips_eps2, label='\u03B5 = 0.013')
plt.plot(amount_of_targets, flips_eps3, label='\u03B5 = 0.02')
# plt.plot(amount_of_targets, flips_eps4)
plt.ylim(0, 80)
# plt.fill_between(amount_of_targets, flips-flip_stds, flips+flip_stds, alpha=0.5)
plt.legend()
plt.ylim(0,30)
plt.show()


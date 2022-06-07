import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

x = np.linspace(-3,10,1000)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def bce(x):
	return -np.log(1-x)

def smart_loss(p, s):
	return (1-0.5*p) * sigmoid(x) + 0.5*p * bce(sigmoid(x))

fig, ax = plt.subplots()
# ax.plot(x, bce(sigmoid(-x)), label='t=1')
# ax.plot(x, bce(sigmoid(x)), label='t=0')
# ax.set_title('BCELoss applied to sigmoid')
ax.plot(x, smart_loss(0,x), label='p = 0', color='g')
ax.plot(x, smart_loss(0.2,x), label='p = 0.2', color='r')
ax.plot(x, smart_loss(0.4,x), label='p = 0.4',color='r')
ax.plot(x, smart_loss(0.5,x), label='p = 0.5',color='r')
ax.plot(x, smart_loss(0.6,x), label='p = 0.6',color='r')
ax.plot(x, smart_loss(0.8,x), label='p = 0.8',color='r')
ax.plot(x, smart_loss(1,x), label='p = 1',color='b')
ax.legend()
plt.show()
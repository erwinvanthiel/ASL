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
ax.plot(x, smart_loss(0,x), label='p = 0')
ax.plot(x, smart_loss(0.2,x), label='p = 0.2')
ax.plot(x, smart_loss(0.4,x), label='p = 0.4')
ax.plot(x, smart_loss(0.5,x), label='p = 0.5')
ax.plot(x, smart_loss(0.6,x), label='p = 0.6')
ax.plot(x, smart_loss(0.8,x), label='p = 0.8')
ax.plot(x, smart_loss(1,x), label='p = 1')
ax.legend()
plt.show()
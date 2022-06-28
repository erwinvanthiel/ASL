import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DecisionBoundary2D():

	def __init__(self, a,b):
		self.a = a
		self.b = b

	def get_orthogonal_distance(self, x, y):
		# line orthogonal to boundary through x,y
		a2 = -1/self.a
		b2 = y - x*a2
		x2 = (b2 - self.b)/(self.a - a2)
		y2 = self.a * x2 + self.b
		#get euclidean distance
		vec, dist = np.array([x2-x,y2-y]), np.sqrt((x2 - x)**2 + (y-y2)**2)
		if y > self.a * x + self.b:
			vec = -1 * vec
			dist = -1 * dist
		return vec, dist

	def f(self, x):
		return self.a * x + self.b


def sigmoid(x):
	return 1 / (1 + np.clip(np.exp(-x), -10000, 10000))

def sigmoid_derivative(x):
	return sigmoid(x) * (1 - sigmoid(x))

def bce_derivative(x):
	return sigmoid(x) 


def step(x,y, decision_boundaries, loss_function_derivative):
	gradient = np.zeros(2)

	for idx, boundary in enumerate(decision_boundaries):
		grad, distance = boundary.get_orthogonal_distance(x,y)
		grad = grad if np.sum(grad ** 2) == 0 else grad / np.sqrt(np.sum(grad ** 2))
		gradient += grad * loss_function_derivative(distance)
	gradient = gradient if np.sum(gradient ** 2) == 0 else gradient / np.sqrt(np.sum(gradient ** 2))

	return np.sign(gradient)


boundary1 = DecisionBoundary2D(1, 2)
boundary2 = DecisionBoundary2D(1, 5.0)
boundary3 = DecisionBoundary2D(-1, 13.0)
boundary4 = DecisionBoundary2D(-1, 16.0)
boundary5 = DecisionBoundary2D(1, 8.0)



xspace = np.linspace(-20, 40)
plt.plot(xspace, boundary1.f(xspace), color='black')
plt.plot(xspace, boundary2.f(xspace), color='black')
plt.plot(xspace, boundary3.f(xspace), color='black')
plt.plot(xspace, boundary4.f(xspace), color='black')
plt.plot(xspace, boundary5.f(xspace), color='black')

a = []
b = []
(x,y) = (8.0,0.0)
for i in range(14):
	gradient = step(x,y, [boundary1, boundary2, boundary3, boundary4, boundary5], bce_derivative)
	a.append(x)
	b.append(y)	
	x += gradient[0]
	y += gradient[1]

plt.plot(a,b, color='blue', label='patient')

a = []
b = []
(x,y) = (8.0,0.0)
for i in range(14):
	gradient = step(x,y, [boundary1, boundary2, boundary3, boundary4, boundary5], sigmoid_derivative)
	a.append(x)
	b.append(y)	
	x += gradient[0]
	y += gradient[1]
	
plt.plot(a,b, color='green', label='greedy')

# PLOT L2 BOUNDS
# r1 = 6
# r2 = 13
# def circle(r, x):
# 	return np.sqrt(r**2 - (x-8)**2)

# circle1x = np.linspace(8 - r1, 8 + r1, 100)
# circle2x = np.linspace(8 - r2, 8 + r2, 100)

# plt.plot(circle1x, circle(r1, circle1x), color='red')
# plt.plot(circle2x, circle(r2, circle2x), color='red')

# PLOT L-infinity BOUNDS
width = 9
plt.gca().add_patch(patches.Rectangle((8 - 0.5*width , 0 - 0.5*width), width, width, linewidth=1, linestyle='--', edgecolor='black',facecolor='none'))
width = 26
plt.gca().add_patch(patches.Rectangle((8 - 0.5*width , 0 - 0.5*width), width, width, linewidth=1, linestyle='--', edgecolor='black',facecolor='none'))

plt.axis([-15, 25, -5, 20])
plt.gca().set_aspect('equal')
plt.legend()

plt.show()



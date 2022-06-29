import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import matplotlib.patches as patches

# Linear decision space
x = np.linspace(0,1)
y = np.linspace(0,1)


x1 = np.linspace(0,1)
y2 = np.linspace(0,1)
z = x1 / x1 * 0.5



x, y = np.meshgrid(x, y)
plane = np.clip(0.5 - x + y, 0, 1)

fig = plt.figure()

ax = fig.gca(projection='3d')


surf = ax.plot_surface(x, y, plane, alpha=0.4, label='confidence plane')
ax.plot(x1,x1,z, color='red', label='decision boundary')
ax.plot(x1,x1,z*0, color='black', label='decision boundary projection')
ax.set_zlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)

surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d

ax.legend()
plt.show()


# Non-linear decision space
x = np.linspace(0,1)
y = np.linspace(0,1)


x1 = np.linspace(0,1)
y2 = np.linspace(0,1)
z = x1 / x1 * 0.5



x, y = np.meshgrid(x, y)
plane = np.clip(0.5 - x**5 + y, 0, 1)

fig = plt.figure()

ax = fig.gca(projection='3d')


ax.plot_surface(x, y, plane, alpha=0.4)
ax.plot(x1**(1/5), x1,z, color='red')
ax.plot(x1**(1/5), x1,z * 0, color='black')


ax.set_zlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
plt.show()
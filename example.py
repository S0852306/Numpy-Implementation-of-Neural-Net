import numpy as np
import matplotlib.pyplot as plt

def f_model(x, theta):
    y = theta[0] * x + theta[1]
    return y

def cost(y, p, theta):
    num_data = len(p)
    J = 0
    for i in range(num_data):
        p = f_model(x[i], theta)
        J += (y[i] - p)**2

    return J

def grad(y, p, theta):

    delta = 0.001
    theta0 = theta
    dx = cost(y, p, theta)
    theta[0] = theta[0] + delta
    
    dx1 = cost(y, p, theta)
    diff1 = (dx1 - dx) / delta

    theta = theta0
    theta[1] = theta[1] + delta

    dx2 = cost(y, p, theta)
    diff2 = (dx2 - dx) / delta

    g = [diff1, diff2]

    return g

theta0 = [3, 2]
theta = theta0
x = np.linspace(0, 1, 20)
y = 4 * x + 3

x = x.tolist()
y = y.tolist()

num_iteration = 10
s0 = 0.01
p = [0] * len(x)
for i in range(num_iteration):
    for j in range(len(x)):
        p[j] = f_model(x[j], theta)
    g = grad(y, p, theta)

    theta[0] = theta[0] - s0 * g[0]
    theta[1] = theta[1] - s0 * g[1]
    J = cost(y, p, theta)
    print(J)

plt.scatter(x, y)
plt.plot(x, p)
plt.show()

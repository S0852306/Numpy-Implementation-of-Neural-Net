from MyNet import FirstOrderSolver
from MyNet import QuasiNewtonSolver
from MyNet import NeuralNet
import matplotlib.pyplot as plt
import numpy as np
NN=NeuralNet([1,5,5,4,1],'tanh')
x=np.array([np.linspace(0,2*np.pi,1000)])
y=x*np.sin(x)+np.cos(3*x)
NN=FirstOrderSolver(x,y,'ADAM',MaxIter=200,StepSize=2e-3,BatchSize=100,Net=NN)
NN=QuasiNewtonSolver(x,y,NN,400)
y0=NN.Evaluate(x)
plt.figure
plt.plot(x[0,:],y0[0,:])
plt.plot(x[0,:],y[0,:])
plt.show()

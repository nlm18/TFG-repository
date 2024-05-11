import numpy as np
import math, random
import matplotlib.pyplot as plt
#%matplotlib inline
from IPython.display import display, Math, Latex

#Creamos la función sigmoide con lambda , donde lambda[0] es la sigmoide y lambda[1] es su derivada
sigm = (lambda x:1/(1+np.e**(-x)),lambda x:x * (1-x))

#Latex
display(Math(r'sigmoide(x) = \frac{1}{1+e^{-x}} '))
display(Math(r"$sigmoide'(x) = x{(1-x)}$"))

v = np.linspace(-5,5,100)
plt.figure(1,figsize=(15,10))
plt.subplot(221)
plt.plot(v,sigm[0](v))
plt.title("Función Sigmoide", fontsize = 15)
plt.text(1, 0.2, ' $\sigma(x) = {1}/(1+e^{-x})$', fontsize = 14, bbox = {'facecolor': 'white', 'alpha': 0.5, 'boxstyle': "square,pad=0.3", 'ec': 'black'})
plt.subplot(222)
plt.plot(v,sigm[1](v), "--", color="red")
plt.title("Derivada función Sigmoide", fontsize = 15)
plt.text(1, -25, '$\sigma\'(x) = x{(1-x)}$', fontsize = 14, bbox = {'facecolor': 'white', 'alpha': 0.5, 'boxstyle': "square,pad=0.3", 'ec': 'black'})
plt.savefig('sigmoide.png')
plt.show()

#Relu Rectified Lineal Unit
relu = (lambda x: np.maximum(0,x), lambda x: 1. * (x > 0))
#Latex
display(Math(r'relu(x) = \max(0,x) '))
display(Math(r"'relu(x) = 1.(x>0) "))

v = np.linspace(-5,5,100)
plt.figure(1,figsize=(15,10))
plt.subplot(221)
plt.plot(v,relu[0](v))
plt.title("Función ReLU", fontsize = 15)
plt.text(-4.5, 4, '$relu(x) = \max(0,x)$', fontsize = 14, bbox = {'facecolor': 'white', 'alpha': 0.5, 'boxstyle': "square,pad=0.3", 'ec': 'black'})
plt.subplot(222)
plt.plot(v,relu[1](v), "--", color="red")
plt.title("Derivada función ReLU", fontsize = 15)
plt.text(-4.5, 0.8, '$relu\'(x) = 1.(x>0)$', fontsize = 14, bbox = {'facecolor': 'white', 'alpha': 0.5, 'boxstyle': "square,pad=0.3", 'ec': 'black'})
plt.savefig('relu.png')
plt.show()
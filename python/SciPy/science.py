import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Creating a function to model and create data
def gerade(x, a, b):
    return a*x+b

def parabel(x, a, b):
    return a*x*x+b

func = parabel

# Generating clean data
x = np.linspace(-3, 3, 100)


y = func(x, 1, -2)

# Adding noise to the data
yn = y + .5 * np.random.normal(size=len(x))

# Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn)

# popt returns the best fit values for parameters of the given model (func).
print(popt)

fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(x,yn)
ax.plot(x,func(x,1,-2))
ax.plot(x,func(x,popt[0],popt[1]))
ax.set_xlabel("x")
ax.set_ylabel("func")
plt.show()

from scipy.optimize import fsolve
solution = fsolve(lambda x: func(x,1,-2),0)
print (solution)
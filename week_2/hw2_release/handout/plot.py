import numpy as np
import sys
import math
import matplotlib.pyplot as plt

x = np.linspace(0, 4*np.pi, 1000)
sin_x = np.sin(x)
cos_x = np.cos(x)
sqrt_X = x**0.5

plt.plot(x, sin_x)
plt.plot(x, cos_x)
plt.plot(x, sqrt_X)

plt.title("IMLD test plot")
plt.legend(["sin_x","cos_x","sqrt_x"])

plt.show()
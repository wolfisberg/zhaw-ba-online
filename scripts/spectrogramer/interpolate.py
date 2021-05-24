import random
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


x = np.arange(0, 11)
y = [random.randrange(0, y) for y in np.arange(10, 21) ** 3]
f = interpolate.interp1d(x, y)
x2 = np.arange(0, 10.1, 2)
y2 = f(x2)

fig, ax = plt.subplots(sharex="all", sharey="all")
plt.plot(x, y, "green")
plt.plot(x2, y2, "red")
plt.show()
print("done")

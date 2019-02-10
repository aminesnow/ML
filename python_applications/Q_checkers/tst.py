import matplotlib.pyplot as plt
import numpy as np


plt.ion()
x, y, z = [], [], []
plt.xlabel('neps')
plt.ylabel('agt 1 error')
line1, line2, = plt.plot(x, y, x, z)
plt.xlim(0, 1000)
plt.ylim(0, 1000)

for i in range(200):
    y.append(2*i)
    z.append(0.5*i)
    x.append(i)
    line1.set_xdata(x)
    line1.set_ydata(y)
    line2.set_xdata(x)
    line2.set_ydata(z)
    plt.draw()
    plt.pause(0.00000001)

plt.waitforbuttonpress()

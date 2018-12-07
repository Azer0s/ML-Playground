import numpy as np
import matplotlib.pyplot as plt

points = [[245, 1400],
          [312, 1600],
          [279, 1700],
          [308, 1875],
          [199, 1350],
          [219, 1550],
          [405, 2350],
          [324, 1780],
          [319, 1600],
          [255, 1700]]

xa = [245,
    312,
    279,
    308,
    199,
    219,
    405,
    324,
    319,
    255]

ya = [1400,
    1600,
    1700,
    1875,
    1350,
    1550,
    2350,
    1780,
    1600,
    1700]
plt.plot(xa, ya, 'ro')

weight = 10
bias = 100
lr = 0.000001
errorAvg = 0

for _ in range(10):
    errorAvg = 0
    for x, y in points:
        prediction = x * weight + bias
        error = y - prediction
        errorAvg += error
        gradient = x * error * lr

        bias = bias + gradient
        weight = weight + weight * gradient
    errorAvg = errorAvg / len(xa)

print(weight, bias, errorAvg)

linex = []
liney = []
for i in range(len(xa)):
    linex += [xa[i]]
    liney += [(weight * xa[i] + bias)]

plt.plot(linex,liney)
plt.show()

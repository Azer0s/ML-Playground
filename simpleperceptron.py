import matplotlib.pyplot as plt


# Helper function to separate the columns of our training data
def column(matrix, i):
    return [row[i] for row in matrix]


# Our training data
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

xa = column(points, 0)
ya = column(points, 1)

weight = 10
bias = 100
lr = 0.000001
errorAvg = 0


def predict(xp):
    return xp * weight + bias


# Train
for iteration in range(10):
    for x, y in points:
        prediction = predict(x)
        error = y - prediction
        errorAvg += error
        gradient = x * error * lr

        # Change bias and weight
        bias += gradient
        weight += weight * gradient
    errorAvg = errorAvg / len(xa)
    print("Iteration: {}\n--------------\nError: {}; Weight: {}; Bias: {}\n".format(iteration, errorAvg, weight, bias))

print("Final\n--------------\nError: {}; Weight: {}; Bias: {}".format(errorAvg,weight, bias))

lineX = []
lineY = []
for i in range(len(xa)):
    lineX += [xa[i]]
    lineY += [predict(xa[i])]

plt.plot(xa, ya, 'ro')
plt.plot(lineX, lineY)
plt.show()

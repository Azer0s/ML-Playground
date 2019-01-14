from random import random


class Perceptron:
    def __init__(self, nrOfInputs):
        self.weight = []
        for i in range(nrOfInputs):
            self.weight += [random()]

    def predict(self, input):
        prediction = 0
        for i, val in enumerate(input):
            prediction += self.weight[i] * val

        return prediction

    def train(self, inputs, outputs, lr, ti):
        errorAvg = 0
        for it in range(ti):
            for input_i, input in enumerate(inputs):
                prediction = self.predict(input)
                error = outputs[input_i] - prediction
                errorAvg += error

                for val_i, val in enumerate(input):
                    gradient = val * error * lr
                    self.weight[val_i] += self.weight[val_i] * gradient

            errorAvg = errorAvg / len(inputs)
        return errorAvg, self.weight


def generate(_in, i, o, ti=1000000, lr=0.001):
    perceptron = Perceptron(_in)
    err, weights = perceptron.train(i, o, lr, ti)
    print("Error: {}; Weights: {}".format(err, weights))
    return perceptron


# Test with two inputs; (input + input)*2 = result
print("Two inputs:")
inputs = [[1, 1], [2, 3], [5, 8], [13, 21], [4, 4], [5, 5], [4, 8]]
outputs = [4, 10, 26, 68, 16, 20, 24]
print(generate(2, inputs, outputs).predict([10, 10]))

# Test with three inputs; (input + input)*2+input = result
print("\nThree inputs:")
inputs = [[1, 1, 1], [2, 3, 4], [5, 8, 9], [13, 21, 22], [4, 4, 5], [5, 5, 6], [4, 8, 9]]
outputs = [5, 14, 35, 90, 21, 26, 33]
print(generate(3, inputs, outputs).predict([10, 10, 5]))

# Test with three inputs; (input + input)*2+input+input*0.1 = result
print("\nFour inputs:")
inputs = [[1, 1, 1, 1], [2, 3, 4, 5], [5, 8, 9, 9], [13, 21, 22, 7], [4, 4, 5, 6], [5, 5, 6, 7], [4, 8, 9, 3]]
outputs = [5.1, 14.5, 35.9, 90.7, 21.6, 26.7, 33.3]
print(generate(4, inputs, outputs).predict([10, 10, 5, 4]))
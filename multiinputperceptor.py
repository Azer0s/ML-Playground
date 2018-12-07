from random import random

class Perceptor:
    def __init__(self, nrOfInputs):
        self.weight = []
        self.lastInput = []
        for i in range(nrOfInputs):
            self.weight += [random()]

    def predict(self, input):
        self.lastInput = input
        prediction = 0
        for i, val in enumerate(input):
            prediction += self.weight[i] * val

        return prediction

    def train(self, inputs, outputs, lr):
        for it in range(100000):
            errorAvg = 0
            for input_i, input in enumerate(inputs):
                prediction = self.predict(input)
                error = outputs[input_i] - prediction
                errorAvg += error

                for val_i, val in enumerate(input):
                    gradient = val * error * lr
                    self.weight[val_i] += self.weight[val_i] * gradient

            errorAvg = errorAvg / len(inputs)
            #print("Iteration: {}; Error: {}".format(it, errorAvg))


#Test with two inputs; (input + input)*2 = result
print("Two inputs:")
inputs = [[1, 1], [2, 3], [5, 8], [13, 21], [4, 4], [5, 5], [4, 8]]
outputs = [4, 10, 26, 68, 16, 20, 24]
perceptor = Perceptor(2)
perceptor.train(inputs, outputs, 0.001)
print(perceptor.weight)
print(perceptor.predict([10, 10]))

#Test with three inputs; (input + input)*2+input = result
print("\nThree inputs:")
inputs = [[1, 1, 1], [2, 3, 4], [5, 8, 9], [13, 21, 22], [4, 4, 5], [5, 5, 6], [4, 8, 9]]
outputs = [5, 14, 35, 90, 21, 26, 33]
perceptor = Perceptor(3)
perceptor.train(inputs, outputs, 0.001)
print(perceptor.weight)
print(perceptor.predict([10, 10, 5]))

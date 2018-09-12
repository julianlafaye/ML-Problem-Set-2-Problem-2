import matplotlib.pyplot as plt
import numpy as np
import random


# Make a prediction with weights
def predict(row, weights):
    # bias = weights[0]
    bias = weights[0]
    # sum(weight[i] * x[i]) + bias = 0 return 1 else 0
    # https://en.wikipedia.org/wiki/Perceptron
    for i in range(len(row) - 1):
        bias += weights[i + 1] * row[i]
    return 1.0 if bias >= 0.0 else 0.0


# Estimate Perceptron weights using stochastic gradient descent
def training(data, l_rate, n):
    # start weights at random numbers
    r = np.random.random_sample()
    weights = [r for i in range(len(data))]
    for t in range(n):
        for row in data:
            prediction = predict(row, weights)
            # error = last row - yes or no value
            # should spit out a negative or positive value if incorrect and 0 if correct
            # https://en.wikipedia.org/wiki/Perceptron
            error = row[-1] - prediction
            # update bias
            weights[0] = weights[0] + l_rate * error
            # for columns in row - the bias
            for i in range(len(row) - 1):
                # update weights
                # a = a + eta(y-yhat)*x
                    # https://en.wikipedia.org/wiki/Perceptron
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        print('>t=%d' % (t))
    return weights


# load data and set learning and n values
filename = 'perceptron_data.csv'
np.genfromtxt(filename, delimiter=',')
dataset = np.genfromtxt(filename, delimiter=',')
print(dataset)
# learning rate
l_rate = 0.5
# number of times
n = 10000
# plot
plt.scatter(dataset[:, 0], dataset[:, 1], c=dataset[:, 2])
weight = training(dataset, l_rate, n)
x = np.linspace(0, 2, 1000)
# w1*x + w2*y + bias = 0
# I don't know if this is the right line, but I have spent a lot of time on the rest and it works.
y = (weight[0] * x + weight[1]) / weight[2]
plt.plot(x, y)
plt.show()

print(weight)

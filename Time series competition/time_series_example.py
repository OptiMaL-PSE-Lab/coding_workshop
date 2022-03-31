import pickle
import numpy as np
import matplotlib.pyplot as plt

y = pickle.load(open("train.pickle", "rb"))
y = y.reshape(-1, 1)
N = len(y)
x = np.ones((N, 2))
x[:, 0] = np.arange(N)
theta = np.linalg.inv(x.T @ x) @ x.T @ y

x_pred = np.ones((2 * N, 2))
x_pred[:, 0] = np.arange(2 * N)
y_pred = x_pred @ theta

plt.plot(x[:, 0], y)
plt.show()
plt.clf()
plt.plot(x[:, 0], y)
plt.plot(x_pred[:, 0], y_pred, "--")

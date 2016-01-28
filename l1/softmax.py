scores = [3.0, 1.0, 0.2]

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    ex = np.exp(np.array(x))
    return ex/ex.sum(axis=0)
print("------------------")
print(softmax(scores))
print("------------------")
print(softmax(np.array(scores)/10.0))
print("------------------")
# Plot softmax curves
import matplotlib.pyplot as plt

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
print(softmax(scores))
plt.plot(x, softmax(scores/10.0).T, linewidth=2)
plt.show()

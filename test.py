import os
import math as m
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as spi
import numpy as np


def integrand(x, tau):
    return np.exp(-x / tau)


tau = 10.0  # Replace with the desired value for tau
lower_limit = 70.0
upper_limit = 110.0

result, error = spi.quad(integrand, lower_limit, upper_limit)
print("Integration result:", result)
print("Error estimate:", error)

data = [1, 2, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 7, 8, 9]
n, bin, _ = plt.hist(
    data, bins=10, range=[0, 10], edgecolor="black", label="Signal", alpha=0.5
)  # hidden on histogram
# mode_index = n.argmax()
# print(sum(n[5:7]))
# print(bin)
# print(mode_index)
# print(n[mode_index])
# print(len(bin))

# steps = np.linspace(0, 5, 5000)
# print(steps)

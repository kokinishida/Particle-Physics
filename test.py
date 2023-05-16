import os
import math as m
import numpy as np
from matplotlib import pyplot as plt

data=[1,2,2,2,3,3,4,5,6,6,6,6,6,6,6,7,8,9]
n,bin,_ = plt.hist(data, bins=10, range=[0,10], edgecolor='black', label='Signal',alpha=0.5) # hidden on histogram
mode_index = n.argmax()
print(sum(n[5:7]))
print(bin)
print(mode_index)
print(n[mode_index])
print(len(bin))

for i in range(1,(len(bin)-1)/2):
    print(i)

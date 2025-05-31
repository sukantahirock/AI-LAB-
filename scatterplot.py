# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:08:56 2023

@author: USER
"""

import matplotlib.pyplot as plt
import numpy as np
data = [10,11,12,13,17,19,23,25,26,21,29,24,22,15,17,14,20,28,30,21]
plt.hist(data, bins=10, edgecolor='black')

std_deviation = np.std(data)
variance = np.var(data)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram Example')

print(f"Standard Deviation: {std_deviation:.2f}")
print(f"Variance: {variance:.2f}")

plt.show()
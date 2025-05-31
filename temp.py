        # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
data = [1,2,3,4,5,6,7,13,15,16,17,18,19,22,24,27,38,39,32,35]
plt.boxplot(data)

std_deviation = np.std(data)
variance = np.var(data)
plt.ylabel('Values')
plt.title('Boxplot Example')
print(f"Standard Deviation: {std_deviation:.2f}")
print(f"Variance: {variance:.2f}")
plt.show()
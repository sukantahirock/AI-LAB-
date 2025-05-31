# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 15:16:53 2023

@author: USER
"""

import numpy as np
data = np.array([10,23,29,34,35,38,21,25,34,36,31,28,29,24,22,11,18,13,13,23])
std_deviation = np.std(data)
variance = np.var(data)
print(f"Standard Deviation: {std_deviation:.2f}")
print(f"Variance: {variance:.2f}")
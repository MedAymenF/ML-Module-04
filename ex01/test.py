#!/usr/bin/env python3
import numpy as np
from l2_reg import iterative_l2, l2

x = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
# Example 1:
print(iterative_l2(x))
# Output:
# 911.0

# Example 2:
print(l2(x))
# Output:
# 911.0

y = np.array([[3], [0.5], [-6]])
# Example 3:
print(iterative_l2(y))
# Output:
# 36.25

# Example 4:
print(l2(y))
# Output:
# 36.25

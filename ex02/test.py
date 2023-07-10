#!/usr/bin/env python3
import numpy as np
from linear_loss_reg import reg_loss_

y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
y_hat = np.array([[3], [13], [-11.5], [5], [11], [5], [-20]])
theta = np.array([[1], [2.5], [1.5], [-0.9]])

# Example 1:
print(reg_loss_(y, y_hat, theta, .5))
# Output:
# 0.8503571428571429

# Example 2:
print(reg_loss_(y, y_hat, theta, .05))
# Output:
# 0.5511071428571429

# Example 3:
print(reg_loss_(y, y_hat, theta, .9))
# Output:
# 1.116357142857143

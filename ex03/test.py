#!/usr/bin/env python3
import numpy as np
from logistic_loss_reg import reg_log_loss_

y = np.array([[1], [1], [0], [0], [1], [1], [0]])
y_hat = np.array([[.9], [.79], [.12], [.04], [.89], [.93], [.01]])
theta = np.array([[1], [2.5], [1.5], [-0.9]])

# Example 1:
print(reg_log_loss_(y, y_hat, theta, .5))
# Output:
# 0.40824105118138265

# Example 2:
print(reg_log_loss_(y, y_hat, theta, .05))
# Output:
# 0.10899105118138264

# Example 3:
print(reg_log_loss_(y, y_hat, theta, .9))
# Output:
# 0.6742410511813826

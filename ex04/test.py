#!/usr/bin/env python3
import numpy as np
from reg_linear_grad import reg_linear_grad, vec_reg_linear_grad

x = np.array([
    [-6, -7, -9],
    [13, -2, 14],
    [-7, 14, -1],
    [-8, -4, 6],
    [-5, -9, 6],
    [1, -5, 11],
    [9, -11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])

# Example 1.1:
print(repr(reg_linear_grad(y, x, theta, 1)))
# Output:
'''
array([[-60.99 ],
[-195.64714286],
[863.46571429],
[-644.52142857]])
'''

# Example 1.2:
print(repr(vec_reg_linear_grad(y, x, theta, 1)))
# Output:
'''
array([[-60.99 ],
[-195.64714286],
[863.46571429],
[-644.52142857]])
'''

# Example 2.1:
print(repr(reg_linear_grad(y, x, theta, 0.5)))
# Output:
'''
array([[-60.99 ],
[-195.86142857],
[862.71571429],
[-644.09285714]])
'''

# Example 2.2:
print(repr(vec_reg_linear_grad(y, x, theta, 0.5)))
# Output:
'''
array([[-60.99 ],
[-195.86142857],
[862.71571429],
[-644.09285714]])
'''

# Example 3.1:
print(repr(reg_linear_grad(y, x, theta, 0.0)))
# Output:
'''
array([[-60.99 ],
[-196.07571429],
[861.96571429],
[-643.66428571]])
'''

# Example 3.2:
print(repr(vec_reg_linear_grad(y, x, theta, 0.0)))
# Output:
'''
array([[-60.99 ],
[-196.07571429],
[861.96571429],
[-643.66428571]])
'''

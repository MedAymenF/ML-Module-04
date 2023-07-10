#!/usr/bin/env python3
import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])

print("Logistic Regression:")
mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09], penalty='none')

# Example 0:
predictions = mylr.predict_(X)
print(repr(predictions))
# Output:
'''
array([[0.99930437],
[1. ],
[1. ]])
'''

# Example 1:
print(mylr.loss_(Y, predictions))
# Output:
# 11.513157421577004

# Example 2:
mylr.fit_(X, Y)
print(repr(mylr.thetas))
# Output:
'''
array([[ 1.04565272],
[ 0.62555148],
[ 0.38387466],
[ 0.15622435],
[-0.45990099]])
'''

# Example 3:
predictions = mylr.predict_(X)
print(repr(predictions))
# Output:
'''
array([[0.72865802],
[0.40550072],
[0.45241588]])
'''

# Example 4:
print(mylr.loss_(Y, predictions))
# Output:
# 0.5432466580663214


print("\nRegularized Logistic Regression:")
mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09], penalty='l2')

# Example 0:
predictions = mylr.predict_(X)
print(repr(predictions))
# Output:
'''
array([[0.99930437],
[1. ],
[1. ]])
'''

# Example 1:
print(mylr.loss_(Y, predictions))
# Output:
# 11.513157421577004

# Example 2:
mylr.fit_(X, Y)
print(repr(mylr.thetas))
# Output:
'''
array([[ 1.04565272],
[ 0.62555148],
[ 0.38387466],
[ 0.15622435],
[-0.45990099]])
'''

# Example 3:
predictions = mylr.predict_(X)
print(repr(predictions))
# Output:
'''
array([[0.72865802],
[0.40550072],
[0.45241588]])
'''

# Example 4:
print(mylr.loss_(Y, predictions))
# Output:
# 0.5432466580663214

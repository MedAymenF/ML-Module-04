#!/usr/bin/env python3
import numpy as np
from ridge import MyLinearRegression as MyLR, MyRidge as MyR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])

print("Linear Regression:")
mylr = MyLR([[1.], [1.], [1.], [1.], [1]])

# Example 0:
y_hat = mylr.predict_(X)
print(repr(y_hat))
# Output:
# array([[8.], [48.], [323.]])

# Example 1:
print(repr(mylr.loss_elem_(Y, y_hat)))
# Output:
# array([[225.], [0.], [11025.]])

# Example 2:
print(mylr.loss_(Y, y_hat))
# Output:
# 1875.0

# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(repr(mylr.thetas))
# Output:
# array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])

# Example 4:
y_hat = mylr.predict_(X)
print(repr(y_hat))
# Output:
# array([[23.417..], [47.489..], [218.065...]])

# Example 5:
print(repr(mylr.loss_elem_(Y, y_hat)))
# Output:
# array([[0.174..], [0.260..], [0.004..]])

# Example 6:
print(mylr.loss_(Y, y_hat))
# Output:
# 0.0732..


# Ridge Regression
print("\nRegularized Linear Regression:")
myr = MyR([[1.], [1.], [1.], [1.], [1]], lambda_=1)

# Example 0:
y_hat = myr.predict_(X)
print(repr(y_hat))

# Example 1:
print(repr(myr.loss_elem_(Y, y_hat)))

# Example 2:
print(myr.loss_(Y, y_hat))

# Example 3:
myr.alpha = 1.6e-4
myr.max_iter = 200000
myr.fit_(X, Y)
print(repr(myr.get_params_()))

# Example 4:
y_hat = myr.predict_(X)
print(repr(y_hat))

# Example 5:
print(repr(myr.loss_elem_(Y, y_hat)))

# Example 6:
print(myr.loss_(Y, y_hat))

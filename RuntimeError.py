
import numpy as np
import sympy
from sympy import *

np.seterr(all='raise')
x= sympy.Symbol('x')
y = ln(x)
function = lambdify(x, y)
data_x = np.linspace(-10,10,256)
data_y = function(data_x)
print(data_y)
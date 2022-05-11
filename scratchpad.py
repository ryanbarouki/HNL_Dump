import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def I(xu, xd, xl):
    l = lambda a, b, c: a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c
    integrand = lambda x: (x - xl**2 - xd**2)*(1 + xu**2 - x)*np.sqrt(l(x,xl**2,xd**2)*l(1,x,xu**2))/x
    integral = integrate.quad(integrand, (xd + xl)**2, (1-xu)**2)
    return 12*integral[0]

# quick test to check above function is doing what it should
func_vals = []
xus = np.linspace(0.1, 0.5, 25)
for xu in xus:
    func_vals.append(I(xu,0.5,0)/I(xu,0,0))

plt.plot(xus, func_vals)
plt.yscale('log')
plt.xscale('log')
plt.show()
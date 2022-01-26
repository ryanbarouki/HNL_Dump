import numpy as np

class Integrator:
    def __init__(self, limits, func_domain=lambda x: True):
        self.limits = np.array(limits)
        self.lower_lim = self.limits[:,0]
        self.upper_lim = self.limits[:,1]
        self.domain_volume = np.prod(self.upper_lim - self.lower_lim)
        self.func_domain = func_domain
    
    def monte_carlo(self, func, niter=10, n=1000):     
        total = 0
        for i in range(niter):
            total += self.__mc_iteration(func, n)
        return total / niter

    def __mc_iteration(self, func, n):
        dim = len(self.limits)
        x_list = np.random.uniform(self.lower_lim, self.upper_lim, (n, dim))

        inside_domain = [self.func_domain(x) for x in x_list]
        frac_in_domain = sum(inside_domain)/len(inside_domain)
        domain = self.domain_volume * frac_in_domain

        y = func(x_list[inside_domain])
        y_mean = y.sum()/len(y)

        return domain * y_mean

import numpy as np
import matplotlib.pyplot as plt
import vegas
import mc_integrator as mc
import gvar
import time

class Integrand:
    count = 0 # static counter to count how many times points are outside of integration region 
    total = 0
    def __init__(self, gammaCM, betaCM, mM, mN, mL, s, b=0.93e-6):
        # all numerical constants defined here
        self.gammaCM = gammaCM
        self.betaCM = betaCM
        self.mM = mM # meson mass
        self.mN = mN # mass of HNL or whatever is secondary decay
        self.mL = mL # mass of lepton (tau)
        self.s = s
        self.b = b

    def c_thetaM(self, n, eN, c_thetaN, eM, phiM):
        aux0=(np.sqrt(((eN**2)-(self.mN**2))))*(((-2.*(eM*eN))+((self.mM**2)+(self.mN**2)))-(self.mL**2))
        aux1=(eM**2)*(((eN**2)+((c_thetaN**2)*(self.mN**2)))-((c_thetaN**2)*(eN**2)))
        aux2=(-1.+(c_thetaN**2))*(((eM**2)-(self.mM**2))*(((eN**2)-(self.mN**2))*(((np.cos(phiM))**2))))
        aux3=(4.*(eM*(eN*(((self.mL**2)-(self.mN**2))-(self.mM**2)))))+((4.*aux1)+(4.*aux2))
        aux4=(2.*((self.mM**2)*(self.mN**2)))+((-4.*((c_thetaN**2)*((self.mM**2)*(self.mN**2))))+((self.mN**4.)+aux3))
        aux5=(-2.*((self.mL**2)*(self.mM**2)))+((self.mM**4.)+((-2.*((self.mL**2)*(self.mN**2)))+aux4))
        aux6=(((np.cos(phiM))**2))*((self.mL**4.)+((4.*((c_thetaN**2)*((eN**2)*(self.mM**2))))+aux5))
        aux7=(-1.+(c_thetaN**2))*(((eM**2)-(self.mM**2))*(((eN**2)-(self.mN**2))*aux6))
        if aux7 < 0:
            # complex valued
            return None
        aux8=0.5*((c_thetaN*((np.sqrt(((eM**2)-(self.mM**2))))*aux0))+ ((-1)**n)*(np.sqrt(aux7)))
        aux9=(aux8/(((-1.+(c_thetaN**2))*(((np.cos(phiM))**2)))-(c_thetaN**2)))/((eN**2)-(self.mN**2))
        output=aux9/((eM*2)-(self.mM**2))
        return output
        
    def integrand_n(self, n, eN, c_thetaN):
        def partial(y):
            eM = y[0]
            phiM = y[1]
            c_thetaM = self.c_thetaM(n, eN, c_thetaN, eM, phiM)

            # is complex or cos(theta) > 1
            Integrand.total += 1
            if c_thetaM == None or abs(c_thetaM) > 1:
                Integrand.count += 1
                return 0

            aux0=((c_thetaM*(np.sqrt(((eM**2)-(self.mM**2)))))-(eM*self.betaCM))*self.gammaCM
            aux1=(eM*(np.sqrt(((eM**2)-(self.mM**2)))))+(c_thetaM*((self.mM**2)*self.betaCM))
            aux2=np.abs(((self.s**-0.5)*((aux1-(c_thetaM*((eM**2)*self.betaCM)))*self.gammaCM)))
            aux3=c_thetaM*(c_thetaN*((np.sqrt(((eM**2)-(self.mM**2))))*((eN**2)-(self.mN**2))))
            aux4=(np.sqrt(((eM**2)-(self.mM**2))))*(((eN**2)-(self.mN**2))*(np.cos(phiM)))
            aux5=aux3+((np.sqrt((1.-(c_thetaM**2))))*((np.sqrt((1.-(c_thetaN**2))))*aux4))
            aux6=c_thetaN*((np.sqrt(((eM**2)-(self.mM**2))))*(np.sqrt(((eN**2)-(self.mN**2)))))
            aux7=(np.sqrt(((eM**2)-(self.mM**2))))*((np.sqrt(((eN**2)-(self.mN**2))))*(np.cos(phiM)))
            aux8=(c_thetaM*aux6)+((np.sqrt((1.-(c_thetaM**2))))*((np.sqrt((1.-(c_thetaN**2))))*aux7))
            aux9=(aux5-(eM*(eN*(np.sqrt(((eN**2)-(self.mN**2)))))))*((((self.mM**-2.)*(((aux8-(eM*eN))**2)))-(self.mN**2))**-0.5)
            aux10=c_thetaN*((np.sqrt(((eM**2)-(self.mM**2))))*(np.sqrt(((eN**2)-(self.mN**2)))))
            aux11=(np.sqrt(((eM**2)-(self.mM**2))))*((np.sqrt(((eN**2)-(self.mN**2))))*(np.cos(phiM)))
            aux12=(c_thetaM*aux10)+((np.sqrt((1.-(c_thetaM**2))))*((np.sqrt((1.-(c_thetaN**2))))*aux11))
            aux13=((np.sqrt((1.-(c_thetaM**2))))*c_thetaN)-(c_thetaM*((np.sqrt((1.-(c_thetaN**2))))*(np.cos(phiM))))
            aux14=(np.sqrt(((eM**2)-(self.mM**2))))*((np.sqrt(((eN**2)-(self.mN**2))))*aux13)
            aux15=(np.sqrt((1.-(c_thetaN**2))))*(eM*((np.sqrt(((eN**2)-(self.mN**2))))*(np.cos(phiM))))
            aux16=(c_thetaM*(c_thetaN*(eM*(np.sqrt(((eN**2)-(self.mN**2)))))))+((np.sqrt((1.-(c_thetaM**2))))*aux15)
            aux17=(((eM**2)-(self.mM**2))**-0.5)*(aux16-(eN*(np.sqrt(((eM**2)-(self.mM**2))))))
            aux18=(np.sqrt(((eM**2)-(self.mM**2))))*((np.sqrt(((eN**2)-(self.mN**2))))*(np.sin(phiM)))
            aux19=((np.sqrt((1.-(c_thetaM**2))))*((np.sqrt((1.-(c_thetaN**2))))*aux18))/self.mM
            aux20=(((np.abs(((((1.-(c_thetaM**2))**-0.5)*aux14)/self.mM)))**2))+((((np.abs((aux17/self.mM)))**2))+(((np.abs(aux19))**2)))
            aux21=((1.+(-2.*(np.abs(((self.s**-0.5)*aux0)))))**6.)*(aux2*((np.abs((aux9/(aux12-(eM*eN)))))*(aux20**-0.5)))
            output=4.*((np.exp((self.b*((-1.+(c_thetaM**2))*((eM**2)-(self.mM**2))))))*aux21)
            return output
        return partial

    def integrand(self, eN, c_thetaN):
        return lambda y: self.integrand_n(1, eN, c_thetaN)(y) + self.integrand_n(2, eN, c_thetaN)(y)
    
    def integrate(self, eN, c_thetaN, limits, n_iter):
        integ = vegas.Integrator(limits)
        f = lambda y: self.integrand(eN, c_thetaN)([y[0], 0.])
        result = integ(f, nitn=15, neval=n_iter)
        return result.mean

def main():
    integral = Integrand(gammaCM=2, betaCM=0.87, mM=1800, mN=1000, mL=100, s=52000)
    # f = [integral.integrate(eN=1200, c_thetaN=c_thetaN, n_iter=1000, limits=[[1800, 15000]]) for c_thetaN in np.linspace(1, 1 - 1e-4, 100)]
    f = [[integral.integrate(eN=eN, c_thetaN=c_thetaN, n_iter=1000, limits=[[1800, 15000]]) for c_thetaN in np.linspace(1, 1-1e-4, 20)]
        for eN in np.linspace(1001, 5000, 100)]
    print(Integrand.count)
    print(Integrand.total)
    print(Integrand.count/Integrand.total)
    plt.plot(f)
    plt.show() 

if __name__ == "__main__":
    main()

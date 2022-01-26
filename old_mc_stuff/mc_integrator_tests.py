import unittest
import mc_integrator as mc

# python -m unittest -v mc_integrator_tests
class MonteCarloIntegratorTests(unittest.TestCase):
    def test_x_squared(self):
        integ = mc.Integrator([[-1, 1]])
        result = integ.monte_carlo(lambda x: x**2)
        print(result)
        delta = 0.02
        # answer is 2/3 = 0.666...
        # this is a crude guess of the errors but for now it will do
        self.assertTrue(result <= 2/3 + delta)
        self.assertTrue(result >= 2/3 - delta)

    def test_x_squared_y_squared(self):
        integ = mc.Integrator([[-1, 1], [-1, 1]])
        result = integ.monte_carlo(lambda x: (x[:,0]**2)*(x[:,1]**2))
        print(result)
        delta = 0.02
        # answer is 4/9 = 0.444...
        self.assertTrue(result <= 4/9 + delta)
        self.assertTrue(result >= 4/9 - delta)

if __name__ == "__main__":
    unittest.main()
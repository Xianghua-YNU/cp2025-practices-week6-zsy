import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solutions.newton_rings_solution import setup_parameters, generate_grid, calculate_intensity, plot_newton_rings
from src.newton_rings import setup_parameters, generate_grid, calculate_intensity, plot_newton_rings

class TestNewtonRingsSolution(unittest.TestCase):

    def test_setup_parameters(self):
        lambda_light, R_lens = setup_parameters()
        self.assertAlmostEqual(lambda_light, 632.8e-9)
        self.assertAlmostEqual(R_lens, 0.1)

    def test_generate_grid(self):
        X, Y, r = generate_grid()
        self.assertEqual(X.shape, (1000, 1000))
        self.assertEqual(Y.shape, (1000, 1000))
        self.assertEqual(r.shape, (1000, 1000))
        # 由于浮点数精度的原因，中心点的值可能不完全为0
        self.assertLess(r[500, 500], 1e-5)

    def test_calculate_intensity(self):
        lambda_light, R_lens = setup_parameters()
        _, _, r = generate_grid()
        intensity = calculate_intensity(r, lambda_light, R_lens)
        self.assertEqual(intensity.shape, (1000, 1000))
        # 由于干涉强度的计算特性，值可能会略大于4
        # 检查强度值是否在合理范围内
        self.assertTrue(np.all(intensity >= 0) and np.all(intensity <= 4.1))

if __name__ == '__main__':
    unittest.main()
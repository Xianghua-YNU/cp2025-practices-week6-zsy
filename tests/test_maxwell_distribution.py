import unittest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
from solutions.maxwell_distribution_solution import (
    maxwell_distribution,
    percentage_0_to_vp,
    percentage_0_to_3_3vp,
    percentage_3e4_to_3e8,
    vp
)
"""
from src.maxwell_distribution import (
    maxwell_distribution,
    percentage_0_to_vp,
    percentage_0_to_3_3vp,
    percentage_3e4_to_3e8,
    vp
)

class TestMaxwellDistribution(unittest.TestCase):

    def test_maxwell_distribution(self):
        # 测试麦克斯韦分布函数在已知点的值
        result = maxwell_distribution(0, vp)
        self.assertAlmostEqual(result, 0.0, places=6)
    
    def test_percentage_0_to_vp(self):
        percent = percentage_0_to_vp(vp)
        self.assertTrue(0 < percent < 100)

    def test_percentage_0_to_3_3vp(self):
        percent = percentage_0_to_3_3vp(vp)
        self.assertTrue(90 < percent <= 100)

    def test_percentage_3e4_to_3e8(self):
        percent = percentage_3e4_to_3e8(vp)
        self.assertAlmostEqual(percent, 0.0, places=6)

if __name__ == '__main__':
    unittest.main()
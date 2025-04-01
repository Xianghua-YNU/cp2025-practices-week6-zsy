import unittest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solutions.standing_wave_solution import sineWaveZeroPhi, init, animate, lines, x
from src.standing_wave import sineWaveZeroPhi, init, animate, lines, x

class TestStandingWave(unittest.TestCase):
    def test_sineWaveZeroPhi(self):
        '''
        测试波函数 sineWaveZeroPhi 是否正确计算
        '''
        # 测试参数
        A = 1
        omega = 2 * np.pi
        k = np.pi / 2
        x_test = np.array([0, 1, 2])
        t_test = 0

        # 计算预期结果
        expected_result = A * np.sin(k * x_test - omega * t_test)

        # 调用函数得到的结果
        result = sineWaveZeroPhi(x_test, t_test, A, omega, k)

        # 测试实际与预期结果是否相同
        np.testing.assert_array_almost_equal(result, expected_result, decimal=6)

    def test_init(self):
        '''
        测试init函数是否正确初始化动画
        '''
        initialized_lines = init()
        self.assertEqual(len(initialized_lines), 3)

        for line in initialized_lines:
            xdata, ydata = line.get_data()
            self.assertEqual(len(xdata), 0)
            self.assertEqual(len(ydata), 0)

    def test_animate(self):
        '''
        测试animate函数是否正确更新动画数据
        '''
        test_frame = 10
        updated_lines = animate(test_frame)
        self.assertEqual(len(updated_lines), 3)

        # 检查返回的line数据的长度是否符合预期
        for line in updated_lines:
            xdata, ydata = line.get_data()
            self.assertEqual(len(xdata), len(x))
            self.assertEqual(len(ydata), len(x))

        # 检查驻波的y数据是否为两个单独波叠加
        A = 1
        omega = 2 * np.pi
        k = np.pi / 2
        t = 0.01 * test_frame

        expected_y1 = sineWaveZeroPhi(x, t, A, omega, k)
        expected_y2 = sineWaveZeroPhi(x, t, A, -omega, k)
        expected_y3 = expected_y1 + expected_y2

        # 获取动画函数中实际的y数据
        actual_y3 = updated_lines[2].get_data()[1]

        # 验证求和结果
        np.testing.assert_array_almost_equal(actual_y3, expected_y3, decimal=6)

if __name__ == '__main__':
    unittest.main()

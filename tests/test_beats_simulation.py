import pytest
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solutions.beats_simulation_solution import simulate_beat_frequency, parameter_sensitivity_analysis
from src.beats_simulation import simulate_beat_frequency, parameter_sensitivity_analysis

class TestBeatFrequencySimulation:
    """测试拍频模拟功能"""
    
    def test_default_parameters(self):
        """测试默认参数下的拍频计算"""
        t, wave, beat_freq = simulate_beat_frequency(show_plot=False)
        assert beat_freq == 4  # 440Hz和444Hz的差应为4Hz
        assert len(t) == 5000  # 默认采样点数
        assert len(wave) == 5000
        
    def test_custom_parameters(self):
        """测试自定义参数"""
        t, wave, beat_freq = simulate_beat_frequency(
            f1=300, f2=305, A1=0.5, A2=1.5, 
            t_start=0, t_end=2, num_points=10000,
            show_plot=False
        )
        assert beat_freq == 5
        assert len(t) == 10000
        assert np.max(wave) <= 2.0  # 振幅和
        
    def test_beat_frequency_calculation(self):
        """测试不同频率差的拍频计算"""
        test_cases = [
            (100, 101, 1),
            (200, 205, 5),
            (500, 490, 10)
        ]
        for f1, f2, expected in test_cases:
            _, _, beat_freq = simulate_beat_frequency(f1=f1, f2=f2, show_plot=False)
            assert beat_freq == expected

class TestParameterSensitivity:
    """测试参数敏感性分析"""
        
    def test_waveform_properties(self):
        """测试波形基本属性"""
        t, wave, _ = simulate_beat_frequency(f1=440, f2=442, show_plot=False)
        # 验证波形周期性
        autocorr = np.correlate(wave, wave, mode='full')
        assert np.argmax(autocorr) == len(autocorr)//2  # 应有明显自相关峰值

if __name__ == "__main__":
    pytest.main(["-v", "--tb=line"])

import numpy as np
import pytest
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from solutions.spring_block_solution import solve_ode_euler, spring_mass_ode_func, solve_ode_odeint
from src.spring_block import solve_ode_euler, spring_mass_ode_func, solve_ode_odeint

def test_solve_ode_euler():
    """测试欧拉法求解器"""
    step_num = 100
    time_points, position, velocity = solve_ode_euler(step_num)
    
    # 检查返回数组的长度
    assert len(time_points) == step_num + 1
    assert len(position) == step_num + 1
    assert len(velocity) == step_num + 1
    
    # 检查初始条件
    assert position[0] == 0
    assert velocity[0] == 1
    
    # 检查时间步长
    assert np.isclose(time_points[1] - time_points[0], 2 * np.pi / step_num)
    
    # 检查解的周期性（近似），使用绝对和相对容差
    assert np.allclose(position[0], position[-1], rtol=0.2, atol=0.1)
    assert np.allclose(velocity[0], velocity[-1], rtol=0.2, atol=0.1)


def test_spring_mass_ode_func():
    """测试微分方程函数"""
    # 测试几个特定点
    test_cases = [
        ([0, 1], 0, [1, 0]),    # x=0, v=1
        ([1, 0], 0, [0, -1]),   # x=1, v=0
        ([-1, 0], 0, [0, 1]),   # x=-1, v=0
        ([0, -1], 0, [-1, 0]),  # x=0, v=-1
    ]
    
    for state, time, expected in test_cases:
        result = spring_mass_ode_func(state, time)
        assert np.allclose(result, expected)


def test_solve_ode_odeint():
    """测试 odeint 求解器"""
    step_num = 100
    time_points, position, velocity = solve_ode_odeint(step_num)
    
    # 检查返回数组的长度
    assert len(time_points) == step_num + 1
    assert len(position) == step_num + 1
    assert len(velocity) == step_num + 1
    
    # 检查初始条件
    assert np.isclose(position[0], 0)
    assert np.isclose(velocity[0], 1)
    
    # 检查解的周期性，使用绝对和相对容差
    assert np.allclose(position[0], position[-1], rtol=0.05, atol=1e-6)
    assert np.allclose(velocity[0], velocity[-1], rtol=0.05, atol=1e-6)
    
    # 检查能量守恒
    energy = 0.5 * (velocity**2 + position**2)
    assert np.allclose(energy, energy[0], rtol=0.05, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
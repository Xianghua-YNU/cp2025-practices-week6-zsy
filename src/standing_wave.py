import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation

def sineWaveZeroPhi(x, t, A, omega, k):
    '''
    返回位置x和时间t的波函数值
    参数:
    x : 空间位置 (array)
    t : 时间 (float)
    A : 振幅 (float)
    omega : 角频率 (float)
    k : 波数 (float)
    '''
    # TODO: 实现正弦波函数
    # 提示：使用 np.sin() 函数计算 A * sin(kx - ωt)
    pass

# 创建动画所需的 Figure 和 Axes
fig = plt.figure()
subplot = plt.axes(xlim=(0, 10), xlabel="x", ylim=(-2, 2), ylabel="y")

# 创建空的line对象，用于动画显示
line1, = subplot.plot([], [], lw=2)
line2, = subplot.plot([], [], lw=2)
line3, = subplot.plot([], [], lw=2)

# 创建一个line对象列表，便于操作
lines = [line1, line2, line3]

def init():
    '''
    动画初始化函数
    TODO: 清空所有line的数据并返回lines列表
    '''
    # 提示：使用line.set_data([], [])设置空数据
    pass

# 创建空间变量x
x = np.linspace(0, 10, 1000)

def animate(i):
    '''
    动画更新函数
    参数: i - 帧序号，自动递增
    TODO: 计算并更新每一帧的波形数据
    '''
    # 定义波的参数
    A = 1
    omega = 2 * np.pi
    k = np.pi / 2
    t = 0.01 * i

    # TODO: 计算两个方向相反的波
    # 提示：使用sineWaveZeroPhi函数，注意第二个波的omega要取负值
    y1 = None
    y2 = None

    # TODO: 计算驻波（两波之和）
    y3 = None

    # TODO: 更新每个line的数据
    # 提示：使用line.set_data(x, y)设置数据
    # 提示：waveFunctions = [[x, y1], [x, y2], [x, y3]]可以帮助组织数据

    return lines
if __name__ == '__main__':
    # TODO: 创建动画对象并显示
    # 提示：使用animation.FuncAnimation创建动画
    # 提示：使用plt.show()显示动画
    pass


import numpy as np
import matplotlib.pyplot as plt

def simulate_beat_frequency(f1=440, f2=444, A1=1.0, A2=1.0, t_start=0, t_end=1, num_points=5000, show_plot=True):
    """
    任务1: 拍频现象的数值模拟
    参数说明:
        f1, f2: 两个波的频率(Hz)
        A1, A2: 两个波的振幅
        t_start, t_end: 时间范围(s)
        num_points: 采样点数
    """
    # 学生任务1: 生成时间范围
    t = np.linspace(t_start, t_end, num_points)
    
    # 学生任务2: 生成两个正弦波
    wave1 = A1 * np.sin(2 * np.pi * f1 * t)
    wave2 = A2 * np.sin(2 * np.pi * f2 * t)

    # 学生任务3: 叠加两个波
    superposed_wave = wave1 + wave2

    # 学生任务4: 计算拍频
    beat_frequency = abs(f1 - f2)

    # 学生任务5: 绘制图像
    if show_plot:
        plt.figure(figsize=(12, 6))
        
        # 绘制第一个波
        plt.subplot(3, 1, 1)
        # 学生任务6: 完成wave1的绘制
        plt.plot(t, wave1)
        plt.title('Wave 1')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        # 绘制第二个波
        plt.subplot(3, 1, 2)
        # 学生任务7: 完成wave2的绘制
        plt.plot(t, wave2)
        plt.title('Wave 2')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        # 绘制叠加波
        plt.subplot(3, 1, 3)
        # 学生任务8: 完成superposed_wave的绘制
        plt.plot(t, superposed_wave)
        plt.title('Superposed Wave')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.savefig('.github/png/beats_waveform.png')
        plt.show()

    return t, superposed_wave, beat_frequency

def parameter_sensitivity_analysis():
    """
    任务2: 参数敏感性分析
    需要完成:
    1. 分析不同频率差对拍频的影响
    2. 分析不同振幅比例对拍频的影响
    """
    # 学生任务9: 频率差分析
    plt.figure(1, figsize=(12, 8))
    # 学生需要在此处添加频率差分析的代码
    freq_diffs = [1, 2, 5, 10]
    for diff in freq_diffs:
        f1 = 440
        f2 = f1 + diff
        t, _, beat_freq = simulate_beat_frequency(f1, f2, show_plot=False)
        plt.plot(t, beat_freq * np.ones_like(t), label=f'Freq Diff: {diff} Hz')
    
    plt.title('Beat Frequency vs Frequency Difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Beat Frequency (Hz)')
    plt.legend()
    plt.savefig('.github/png/freq_diff_analysis.png')  # 保存为png
    plt.show()
    # 学生任务10: 振幅比例分析
    plt.figure(2, figsize=(12, 8))
    # 学生需要在此处添加振幅比例分析的代码
    amp_ratios = [0.5, 1.0, 2.0]
    for ratio in amp_ratios:
        A1 = 1.0
        A2 = A1 * ratio
        t, _, beat_freq = simulate_beat_frequency(A1=A1, A2=A2, show_plot=False)
        plt.plot(t, beat_freq * np.ones_like(t), label=f'Amp Ratio: {ratio}:1')
    
    plt.title('Beat Frequency vs Amplitude Ratio')
    plt.xlabel('Time (s)')
    plt.ylabel('Beat Frequency (Hz)')
    plt.legend()
    plt.savefig('.github/png/amp_ratio_analysis.png')  # 保存为png
    plt.show()
    
if __name__ == "__main__":
    # 示例调用
    print("=== 任务1: 基本拍频模拟 ===")
    t, wave, beat_freq = simulate_beat_frequency()
    print(f"计算得到的拍频为: {beat_freq} Hz")
    
    print("\n=== 任务2: 参数敏感性分析 ===")
    parameter_sensitivity_analysis()

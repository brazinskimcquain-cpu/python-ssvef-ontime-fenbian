import numpy as np
from scipy.linalg import eigh


class TRCAFilter:
    def __init__(self):
        self.W = None  # 空间滤波权重矩阵

    def train(self, calibration_data):
        """
        离线训练 TRCA 权重
        :param calibration_data: 形状为 (trials, channels, samples) 的 numpy 数组
        """
        trials, channels, samples = calibration_data.shape
        print(f"[TRCA] 开始计算权重... 输入数据: 试次={trials}, 通道={channels}, 采样点={samples}")

        # 计算均值模板 (Target Template)
        mean_data = np.mean(calibration_data, axis=0)
        mean_data -= np.mean(mean_data, axis=1, keepdims=True)  # 去中心化
        S = np.dot(mean_data, mean_data.T)

        # 计算各个独立试次的协方差总和
        Q = np.zeros((channels, channels))
        for i in range(trials):
            trial_data = calibration_data[i, :, :].copy()
            trial_data -= np.mean(trial_data, axis=1, keepdims=True)  # 去中心化
            Q += np.dot(trial_data, trial_data.T)

        # 正则化防止高维矩阵奇异报错
        Q += np.eye(channels) * 1e-6

        # 广义特征值分解，求出让信噪比(SNR)最大的空间方向
        eigenvalues, eigenvectors = eigh(S, Q)

        # 提取最大特征值对应的特征向量作为空间滤波器权重
        self.W = eigenvectors[:, -1]

        print(f"[TRCA] ✓ 权重矩阵提取成功！权重形状: {self.W.shape}")
        return self.W

    def apply(self, realtime_data):
        """
        实时过滤：将多通道脏数据降维、融合成单通道纯净数据
        :param realtime_data: 形状为 (channels, samples) 的实时数组
        """
        if self.W is None:
            # 🎯 核心修复：如果还没经过训练提取到权重，绝对不能求均值！
            # 暴力求均值会导致信号的正负波峰互相抵消，直接抹平 SSVEP 频点能量。
            # 安全的做法是：原样退回原始数据 (Bypass)，由后续的算法自己去处理。
            return realtime_data.copy()

        # 真正的 TRCA 降维提纯：(channels,) 点乘 (channels, samples) -> (1, samples)
        enhanced_signal = np.dot(self.W.T, realtime_data).reshape(1, -1)
        return enhanced_signal
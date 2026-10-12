import torch
import numpy as np
import scipy.signal as signal
from model import EEGNet


class MEGClassificationModule:
    def __init__(self, model_path, device='cpu', in_channels=59):
        self.device = torch.device(device)

        # ---------------------------------------------------------
        # 核心参数：严格锁定训练时的物理标尺
        # ---------------------------------------------------------
        self.train_fs = 1024.0  # 训练时的采样率
        self.train_n_points = 4609  # 训练时的时间点数
        self.signal_length = 167  # 网络最终接收的特征数

        # 💡 【核心修改一】：实例化模型时，强行适配 59 通道，杜绝 size mismatch
        self.model = EEGNet(in_channels=in_channels, signal_length=self.signal_length).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # 预计算训练时使用的频率刻度
        self.freqs = torch.fft.rfftfreq(self.train_n_points, d=1 / self.train_fs)
        self.valid_idx = torch.where((self.freqs >= 3.0) & (self.freqs <= 40.0))[0]

        self.classes = {0: "9Hz", 1: "14Hz"}

    def preprocess(self, raw_data, current_fs):
        # 💡 【核心修改二】：提取 59 个真正的跨百人交集健康通道
        target_indices = [
            181, 187, 188, 191, 194, 199, 200, 202, 203, 206, 208, 209, 211, 214, 215,
            217, 218, 220, 221, 226, 227, 230, 232, 233, 235, 236, 239, 241, 242, 244,
            245, 248, 251, 253, 256, 257, 259, 260, 262, 263, 265, 266, 268, 271, 272,
            274, 275, 277, 278, 280, 281, 283, 284, 286, 287, 290, 295, 298, 299
        ]

        # 为了防止传入的数据不够 300 个通道导致索引越界，做一个简单的安全切片
        max_idx_needed = max(target_indices)
        if raw_data.shape[0] <= max_idx_needed:
            print(
                f"[警告] 输入的实时数据通道数不足！需要至少 {max_idx_needed + 1} 个通道，当前只有 {raw_data.shape[0]} 个。")
            # 这是一个毁灭性错误，直接抛出异常，防止模型卡死
            raise IndexError("实时数据通道数与预训练提取索引不匹配！")

        x_np = raw_data[target_indices, :].copy()

        # =========================================================
        # 致命问题修复：强行拉升数值精度 (防吞噬)！
        # 不论是本地 Tesla 还是线上 TCP 传来的 Tesla，这里统统放大到 fT。
        # 让 std 变成几百，从而完全无视 (std + 1e-6) 里的 1e-6，恢复完美的 Z-score！
        # =========================================================
        if np.max(np.abs(x_np)) < 1e-3:
            x_np = x_np * 1e15

        # =========================================================
        # 强行恢复去线性漂移 (对抗线上裸数据的 0Hz 伪影)
        # =========================================================
        x_np = signal.detrend(x_np, axis=1, type='linear')

        # ---------------------------------------------------------
        # 频率分辨率对齐 (时间域重采样与边缘补齐)
        # ---------------------------------------------------------
        # 步骤 A：对齐采样率物理时间
        if current_fs != self.train_fs and current_fs > 0:
            num_target_points = int((x_np.shape[1] / current_fs) * self.train_fs)
            x_np = signal.resample(x_np, num_target_points, axis=1)

        # 步骤 B：将补零 (constant=0) 改为边缘平滑延伸 (mode='edge')
        # 彻底解决强行补 0 造成的低频断崖式伪影！
        if x_np.shape[1] < self.train_n_points:
            pad_width = self.train_n_points - x_np.shape[1]
            x_np = np.pad(x_np, ((0, 0), (0, pad_width)), mode='edge')
        elif x_np.shape[1] > self.train_n_points:
            x_np = x_np[:, :self.train_n_points]

        # 转为 Tensor
        x = torch.tensor(x_np, dtype=torch.float32)

        # ---------------------------------------------------------
        # Z-Score 与 频域特征提取
        # ---------------------------------------------------------
        # 此时 x 已经是 fT 量级，std 会是正常的几十/几百，Z-score 将完美生效！
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x = (x - mean) / (std + 1e-6)

        # FFT 提取原始峰值
        x_fft = torch.fft.rfft(x, dim=1)
        x_amp = torch.abs(x_fft)

        # 提取有效频段
        x_features = x_amp[:, self.valid_idx]

        # 兜底：严丝合缝对齐网络输入维度
        if x_features.shape[1] > self.signal_length:
            x_features = x_features[:, :self.signal_length]
        elif x_features.shape[1] < self.signal_length:
            pad_size = self.signal_length - x_features.shape[1]
            x_features = torch.nn.functional.pad(x_features, (0, pad_size))

        return x_features

    def predict(self, raw_data, fs=1000):
        try:
            features = self.preprocess(raw_data, fs)

            features = features.unsqueeze(0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(features)
                probs = torch.softmax(logits, dim=1)

                prob_14hz = probs[0, 1].item()
                prob_9hz = probs[0, 0].item()

                # 偏袒 14Hz 的阈值判定 (只要 14Hz 特征超过 0.35 就认为是 14Hz)
                THRESHOLD_14HZ = 0.35

                if prob_14hz > THRESHOLD_14HZ:
                    prediction = 1
                    confidence = prob_14hz
                else:
                    prediction = 0
                    confidence = prob_9hz

            return prediction, confidence

        except Exception as e:
            print(f"预测模块出错: {e}")
            return -1, 0.0

    def get_class_name(self, prediction_idx):
        return self.classes.get(prediction_idx, "未知频率")

    def is_confident(self, confidence, threshold=0.6):
        return confidence >= threshold
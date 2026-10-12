import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, in_channels=64, signal_length=167, num_classes=2):
        """
        专门处理 FFT 频域特征的 CNN 架构。
        这里的 in_channels 会被 train.py 动态传入的 59 覆盖！
        """
        super().__init__()
        self.in_channels = in_channels
        self.signal_length = signal_length

        F1 = 8
        D = 2
        F2 = D * F1

        kernel_size_1 = (1, 5)
        # 💡 这里自动适配 59，贯穿全部共同健康通道
        kernel_size_2 = (self.in_channels, 1)
        kernel_size_3 = (1, 3)
        kernel_size_4 = (1, 1)

        padding_1 = (0, kernel_size_1[1] // 2)
        padding_3 = (0, kernel_size_3[1] // 2)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size_1, padding=padding_1, bias=False),
            nn.BatchNorm2d(F1),

            nn.Conv2d(F1, D * F1, kernel_size_2, groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),

            nn.Dropout2d(0.6),

            nn.Conv2d(D * F1, D * F1, kernel_size_3, padding=padding_3, groups=D * F1, bias=False),
            nn.Conv2d(D * F1, F2, kernel_size_4, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),

            nn.Dropout2d(0.6)
        )

        self.flatten = nn.Flatten()

        # 💡 动态推演维度，保证 59 通道下全连接层绝不崩溃
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.in_channels, self.signal_length)
            dummy_output = self.feature_extractor(dummy_input)
            dense_input_size = dummy_output.numel()

        self.Dense = nn.Linear(dense_input_size, num_classes)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        y = self.feature_extractor(x)
        y = self.flatten(y)
        y = self.Dense(y)
        return y
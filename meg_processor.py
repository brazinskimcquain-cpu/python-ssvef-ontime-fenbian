import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations


class SignalProcessor:
    """
    独立封装的脑磁信号清洗与对齐状态机核心逻辑
    """

    def __init__(self, sampling_rate, window_size, update_speed_ms):
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.update_speed_ms = update_speed_ms

        self.trial_state = "WAITING"
        self.collect_countdown = 0
        self.rest_countdown = 0
        self.target_class = ""

    def clean_trigger(self, trg, threshold):
        """
        【彻底修复漏检测】绝对硬阈值去噪器
        抛弃了致命的 np.median，直接使用绝对电压进行判断。
        只要电平超过 threshold (通常是1.0V)，就强制设为 1.0 (高电平)。
        防止长方波脉冲被中位数计算抵消掉！
        """
        clean = np.zeros_like(trg)
        clean[trg > threshold] = 1.0
        return clean

    def process_fsm(self, data, idx_9hz, idx_14hz, threshold):
        """处理单脉冲对齐有限状态机 (FSM)"""
        trg_9_clean = self.clean_trigger(data[idx_9hz, :].copy(), threshold)
        trg_14_clean = self.clean_trigger(data[idx_14hz, :].copy(), threshold)

        should_predict = False
        ground_truth = "无刺激"
        ui_status_text = "静息态 "

        if self.trial_state == "RESTING":
            self.rest_countdown -= 1
            if self.rest_countdown <= 0:
                self.trial_state = "WAITING"
                ui_status_text = "等待脉冲..."
            else:
                ui_status_text = f"冷却期 ，强制倒计时中... ({self.rest_countdown}s)"

        elif self.trial_state == "COLLECTING":
            self.collect_countdown -= 1
            if self.collect_countdown <= 0:
                should_predict = True
                ground_truth = self.target_class
                self.trial_state = "RESTING"

                # ▼▼▼ 修复：将冷却时间从 5000 缩短为 3000 (3秒) ▼▼▼
                self.rest_countdown = int(3000 / self.update_speed_ms)

                ui_status_text = f"数据收集及对齐完毕！真实刺激: {ground_truth}"
            else:
                collected_sec = (self.collect_countdown * self.update_speed_ms) / 1000.0
                ui_status_text = f"已锁定 {self.target_class}！正在填装脑磁缓冲窗... (还剩 {collected_sec:.1f}s)"

        elif self.trial_state == "WAITING":
            new_points = int(self.sampling_rate * (self.update_speed_ms / 1000.0))
            # 只在最新进来的数据段里找 Trigger，防止重复触发
            recent_9 = trg_9_clean[-new_points:]
            recent_14 = trg_14_clean[-new_points:]

            # 只要最新的片段里出现过 1.0，立刻锁定！
            if np.max(recent_9) > 0.5:
                self.trial_state = "COLLECTING"
                self.collect_countdown = int(self.window_size / (self.update_speed_ms / 1000.0)) - 1
                self.target_class = "9Hz"
                ui_status_text = "捕获 9Hz 起始脉冲信号！正在锁定并组装刺激数据..."
            elif np.max(recent_14) > 0.5:
                self.trial_state = "COLLECTING"
                self.collect_countdown = int(self.window_size / (self.update_speed_ms / 1000.0)) - 1
                self.target_class = "14Hz"
                ui_status_text = "捕获 14Hz 起始脉冲信号！正在锁定并组装刺激数据..."
            else:
                ui_status_text = "实时监听中"

        return trg_9_clean, trg_14_clean, should_predict, ground_truth, ui_status_text

    def filter_display_data(self, data, indices, params):
        """对用于显示的波形执行滤波"""
        int_fs = int(self.sampling_rate)
        visual_data_filtered = []

        for ch_idx in indices:
            if ch_idx < data.shape[0]:
                ch_data = data[ch_idx].copy()
                DataFilter.detrend(ch_data, DetrendOperations.CONSTANT.value)
                if params.get('bandpass_enabled', True):
                    DataFilter.perform_bandpass(ch_data, int_fs, params['lowcut'],
                                                params['highcut'], params['order'],
                                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                if params.get('notch_enabled', False):
                    DataFilter.perform_bandstop(ch_data, int_fs, params['notch'] - 1,
                                                params['notch'] + 1, params['order'],
                                                FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
                visual_data_filtered.append(ch_data)

        return visual_data_filtered
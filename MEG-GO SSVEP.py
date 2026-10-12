# =====================================================================
# 【防爆补丁区】：必须放在第一位！解决 PyQt5 和 PyTorch 的 DLL 冲突
# =====================================================================
import sys
import os
from pathlib import Path
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =====================================================================
# 💡 【核心新增】：PyInstaller 资源动态寻址函数
# =====================================================================
def get_resource_path(relative_path):
    """获取资源文件的绝对路径，完美适配 PyInstaller 打包后的 _MEIPASS 临时目录"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 打包后的 EXE 运行时的临时释放路径
        return os.path.join(sys._MEIPASS, relative_path)
    # 本地未打包开发时的当前路径
    return os.path.join(os.path.abspath("."), relative_path)


visual_bci_path = Path(__file__).parent.parent / "visual-bci"
sys.path.append(str(visual_bci_path))

try:
    from yuchuli_ui import MEGClassificationModule
except ImportError:
    print("Warning: 找不到 yuchuli_ui.py，请确保它在同一目录下！")
    MEGClassificationModule = None

try:
    import serial
except ImportError:
    print("Warning: 找不到 serial 库，请使用 'pip install pyserial' 安装！硬件控制将被暂时禁用。")
    serial = None

# =====================================================================
# 【常规与自定义模块导入】
# =====================================================================
import argparse
import logging
import numpy as np
import mne
from Read_MEG_HOST import MEGDataReceiver
from scipy.signal import welch
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QDialog

# 导入拆分后的模块
from ui_main_window import MainWindowUI
from connection_dialog import ConnectionDialog
from meg_processor import SignalProcessor


class MEGSystemController(MainWindowUI):
    """
    系统控制器类。继承自界面类，专注处理数据和业务逻辑。
    """

    # ⚠️ 修复：默认截取窗口改为 4.5 秒，严丝合缝对齐训练集
    def __init__(self, host, port, com_port='COM6', window_size=4.5, update_speed_ms=500):
        super().__init__()

        # 💡 【修改 1】：动态加载 layout 布局文件
        self.layout_path = get_resource_path(os.path.join("layout", "simitMEG-all.lout"))
        try:
            self.layout = mne.channels.read_layout(self.layout_path)
            print(f"成功加载布局文件: {self.layout_path}")
        except Exception as e:
            self.layout = None
            print(f"布局加载失败: {e}")

        self.frame_count = 0
        self.playback_mode = False
        self.playback_raw = None
        self.playback_ptr = 0
        self.record_data = []
        self.recording = False

        self.host = host
        self.port = port
        self.dataloader = None
        self.n_channels = 344
        self.sampling_rate = 1000
        self.window_size = window_size
        self.update_speed_ms = update_speed_ms
        self.last_n_channels = -1
        self.connection_start_time = None

        # 统计指示器变量
        self.stat_total_trials = 0
        self.stat_correct = 0
        self.stat_wrong = 0
        self.stat_high_confidence = 0

        # 硬件串口通讯
        self.ser = None
        self.last_serial_cmd = None
        self.last_light_on_time = 0.0

        if serial is not None and com_port and com_port.upper() != 'NONE':
            try:
                self.ser = serial.Serial(com_port, 9600, timeout=1)
                time.sleep(0.5)
                self.ser.write(b'\xAA')
                self.last_serial_cmd = b'\xAA'
                print(f"[*] 已成功连接 {com_port}，并发送唤醒握手包 0xAA")
            except Exception as e:
                print(f"[-] 串口连接失败: {e}")
                self.ser = None

        self.trigger_idx_9hz = 320
        self.trigger_idx_14hz = 321
        self.trg_threshold = 1.0
        self.online_trigger_mapped = False

        self.processor = SignalProcessor(self.sampling_rate, self.window_size, self.update_speed_ms)

        # 精准匹配 59 个跨百人交集健康通道的真实索引 (0-based)
        self.healthy_indices = [
            181, 187, 188, 191, 194, 199, 200, 202, 203, 206, 208, 209, 211, 214, 215,
            217, 218, 220, 221, 226, 227, 230, 232, 233, 235, 236, 239, 241, 242, 244,
            245, 248, 251, 253, 256, 257, 259, 260, 262, 263, 265, 266, 268, 271, 272,
            274, 275, 277, 278, 280, 281, 283, 284, 286, 287, 290, 295, 298, 299
        ]

        self.display_indices = self.healthy_indices[:24]
        self.channel_names = [f"CH{i}" for i in self.display_indices]

        if MEGClassificationModule is not None:
            # 💡 【修改 2】：动态加载 model 模型文件
            model_path = get_resource_path(os.path.join("model", "Final_Best_EEGNet_S12_V2_Acc_80.18.pth"))

            try:
                self.classifier = MEGClassificationModule(
                    model_path=model_path,
                    device='cpu',
                    in_channels=len(self.healthy_indices)  # 动态传入 59
                )
            except TypeError:
                print("\n[警告] MEGClassificationModule 不支持传入 in_channels 参数。")
                self.classifier = MEGClassificationModule(model_path=model_path, device='cpu')
        else:
            self.classifier = None

        self.server_running = False
        self.processing_paused = False
        self.classification_enabled = True

        self.filter_params = {
            'lowcut': 3.0, 'highcut': 15.0, 'notch': 50.0, 'order': 2,
            'bandpass_enabled': True, 'notch_enabled': False
        }

        self.setup_ui()
        self._update_model_status_label()
        self._connect_signals()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(self.update_speed_ms)

    def _connect_signals(self):
        self.btn_server.clicked.connect(self._toggle_server)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_apply_trg.clicked.connect(self._apply_trg_settings)
        self.btn_clear_stats.clicked.connect(self._reset_statistics)
        self.action_load_fif.triggered.connect(self._load_fif)
        self.action_save_fif.triggered.connect(self._save_to_fif)
        self.action_bandpass.triggered.connect(lambda c: self.filter_params.update({'bandpass_enabled': c}))
        self.action_notch.triggered.connect(lambda c: self.filter_params.update({'notch_enabled': c}))
        self.action_enable_class.triggered.connect(lambda c: setattr(self, 'classification_enabled', c))
        self.action_record.triggered.connect(self._toggle_recording)

    def _update_model_status_label(self):
        is_loaded = self.classifier is not None and hasattr(self.classifier,
                                                            'model') and self.classifier.model is not None
        model_status = "模型已加载" if is_loaded else "模型未加载"
        hw_status = " | 硬件已连接(已发送0xAA)" if self.ser and self.ser.is_open else " | 硬件未连接"
        self.model_status_label.setText(f"{model_status}{hw_status}")
        self.model_status_label.setStyleSheet(
            "font-size: 11pt; font-weight: bold; color: green;" if is_loaded else "font-size: 11pt; color: red;")

    def _cleanup(self):
        if self.ser and self.ser.is_open:
            self._send_serial_cmd(b'\xEE', force=True)
            time.sleep(0.1)
            self.ser.write(b'\x0F')
            self.ser.close()

    def _send_serial_cmd(self, cmd, force=False):
        if self.ser and self.ser.is_open:
            if cmd == b'\xEE' and not force:
                if time.time() - self.last_light_on_time < 1.0:
                    return
            elif cmd in [b'\x11', b'\x22']:
                self.last_light_on_time = time.time()

            if cmd != self.last_serial_cmd:
                try:
                    self.ser.write(cmd)
                    self.last_serial_cmd = cmd
                except Exception:
                    pass

    def _apply_trg_settings(self):
        try:
            self.trigger_idx_9hz = int(self.idx9_input.text())
            self.trigger_idx_14hz = int(self.idx14_input.text())
            self.trg_threshold = float(self.thresh_input.text())
            QMessageBox.information(self, "成功",
                                    f"打标参数已更新\n9Hz索引: {self.trigger_idx_9hz}\n14Hz索引: {self.trigger_idx_14hz}\n触发阈值: {self.trg_threshold}")
            self.online_trigger_mapped = True
        except Exception:
            QMessageBox.warning(self, "错误", "请输入有效的数字！")

    def _reset_statistics(self):
        self.stat_total_trials = 0
        self.stat_correct = 0
        self.stat_wrong = 0
        self.stat_high_confidence = 0
        self.stats_label.setText("准确率: 0.0% (0对/0错) | 高置信度(>0.6): 0次")

    def _update_dynamic_channels(self, actual_n_channels):
        self.last_n_channels = actual_n_channels
        if actual_n_channels > max(self.healthy_indices):
            if self.playback_mode and self.playback_raw is not None:
                self.channel_names = [self.playback_raw.ch_names[i] for i in self.display_indices]
            else:
                self.channel_names = [f"MEG{i}" for i in self.display_indices]
        else:
            max_display = min(actual_n_channels, 24)
            self.display_indices = list(range(max_display))
            self.channel_names = [f"CH{i + 1}" for i in self.display_indices]

        for i, pw in enumerate(self.plots):
            if i < len(self.display_indices):
                pw.show()
                self.legends[i].removeItem(self.curves[i])
                self.legends[i].addItem(self.curves[i], self.channel_names[i])
            else:
                pw.hide()

    def _toggle_server(self):
        self.btn_server.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        if not self.server_running:
            try:
                self.playback_mode = False
                self.dataloader = MEGDataReceiver(host=self.host, port=self.port, n_channels=self.n_channels,
                                                  fs=self.sampling_rate)
                self.dataloader.start_server()

                self.server_running = True
                self.connection_start_time = time.time()
                self.online_trigger_mapped = False

                self.processor.trial_state = "WAITING"
                self.processor.collect_countdown = 0
                self.processor.rest_countdown = 0
                self._reset_statistics()

                self.time_label.setText("已连接: 00:00:00")
                self.time_label.setStyleSheet(
                    "font-size: 14pt; font-weight: bold; color: green; padding: 10px; border: 2px dashed green; border-radius: 5px;")
                self.btn_server.setText("彻底结束")
                self.btn_server.setStyleSheet(
                    "font-size: 14pt; font-weight: bold; background-color: #f44336; color: white; border-radius: 10px;")
                self.btn_pause.setEnabled(True)

                self.title.setText("MEG Signal Preview & Real-Time PSD (正在实时监听...)")
                self.title.setStyleSheet("font-size: 20pt; font-weight: bold; margin: 10px; color: black;")
                self.last_n_channels = -1
                self.record_data = []
            except Exception as e:
                QMessageBox.warning(self, "网络错误", f"无法启动监听！\n报错详情: {e}")
        else:
            try:
                if self.dataloader is not None:
                    self.dataloader.stop_server()
            except Exception:
                pass
            finally:
                self.dataloader = None
                self.server_running = False
                self.processing_paused = False
                self.connection_start_time = None

                self.time_label.setText("已连接: --:--:--")
                self.time_label.setStyleSheet(
                    "font-size: 14pt; font-weight: bold; color: #333; padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
                self.btn_server.setText("启动实时监听")
                self.btn_server.setStyleSheet(
                    "font-size: 14pt; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 10px;")
                self.btn_pause.setText("暂停界面")
                self.btn_pause.setStyleSheet(
                    "font-size: 14pt; font-weight: bold; background-color: #FF9800; color: white; border-radius: 10px;")
                self.btn_pause.setEnabled(False)

                self.title.setText("MEG Signal Preview & Real-Time PSD (连接已切断)")
                self.title.setStyleSheet("font-size: 20pt; font-weight: bold; margin: 10px; color: gray;")
                self.prediction_label.setText("静息态 (连接断开)")
                self.prediction_label.setStyleSheet(
                    "font-size: 16pt; font-weight: bold; padding: 10px; border: 2px solid gray; border-radius: 5px; background-color: #f0f0f0;")

                for curve in self.curves: curve.setData([])
                self.psd_curve.setData([], [])
                self.trigger_curve_9hz.setData([])
                self.trigger_curve_14hz.setData([])
                self._send_serial_cmd(b'\xEE', force=True)
                QtWidgets.QApplication.processEvents()

        QtCore.QTimer.singleShot(500, lambda: self.btn_server.setEnabled(True))

    def _toggle_pause(self):
        self.processing_paused = not self.processing_paused
        if self.processing_paused:
            self.btn_pause.setText("继续界面")
            self.btn_pause.setStyleSheet(
                "font-size: 14pt; font-weight: bold; background-color: #2196F3; color: white; border-radius: 10px;")
            self.title.setText("MEG Signal Preview & Real-Time PSD (界面已冻结)")
        else:
            self.btn_pause.setText("暂停界面")
            self.btn_pause.setStyleSheet(
                "font-size: 14pt; font-weight: bold; background-color: #FF9800; color: white; border-radius: 10px;")
            self.title.setText("MEG Signal Preview & Real-Time PSD (正在接收...)")

    def _toggle_recording(self, enabled: bool):
        self.recording = enabled
        if enabled:
            self.record_data = []
            self.action_record.setText('停止存盘')
            QMessageBox.information(self, "提示",
                                    "已开启数据全通道记录！\n录制完成后可通过 '文件 -> 导出已收数据' 保存为 fif 文件。")
        else:
            self.action_record.setText('开始存盘 (保存完整通道)')

    def _load_fif(self):
        if self.server_running and not self.playback_mode:
            QMessageBox.warning(self, "警告", "请先停止实时网络监听，再加载离线文件！")
            return

        fileName, _ = QFileDialog.getOpenFileName(self, "选择离线数据文件", "", "FIF Files (*.fif)")
        if fileName:
            try:
                self.playback_raw = mne.io.read_raw_fif(fileName, preload=True)
                self.sampling_rate = self.playback_raw.info['sfreq']
                self.processor.sampling_rate = self.sampling_rate

                self.playback_mode = True
                self.playback_ptr = 0
                self.server_running = True
                self._reset_statistics()

                ch_names = self.playback_raw.ch_names
                if 'TRG001' in ch_names:
                    self.trigger_idx_9hz = ch_names.index('TRG001')
                    self.idx9_input.setText(str(self.trigger_idx_9hz))
                if 'TRG002' in ch_names:
                    self.trigger_idx_14hz = ch_names.index('TRG002')
                    self.idx14_input.setText(str(self.trigger_idx_14hz))

                self.title.setText(f"正在数据重演 (Playback): {Path(fileName).name}")
                self.btn_server.setText("停止重演")
                self.btn_server.setStyleSheet(
                    "font-size: 14pt; font-weight: bold; background-color: #f44336; color: white; border-radius: 10px;")
                self.btn_pause.setEnabled(True)

                self.last_n_channels = -1
            except Exception as e:
                QMessageBox.critical(self, "加载失败", str(e))

    def _save_to_fif(self):
        if not self.record_data:
            QMessageBox.warning(self, "警告", "缓冲区没有可保存的数据！请先在菜单栏开启【记录缓冲区】")
            return
        try:
            all_raw_data = np.concatenate(self.record_data, axis=1)
            ch_names = [f"MEG_{i:03d}" for i in range(all_raw_data.shape[0])]
            info = mne.create_info(ch_names=ch_names, sfreq=self.sampling_rate, ch_types='mag')
            raw_obj = mne.io.RawArray(all_raw_data, info)

            f_name, _ = QFileDialog.getSaveFileName(self, "保存FIF文件", "", "FIF Files (*.fif);;All Files (*)")
            if f_name:
                if not f_name.endswith('.fif'): f_name += '.fif'
                raw_obj.save(f_name, overwrite=True)
                QMessageBox.information(self, "成功",
                                        f"全通道FIF文件保存成功！\n共保存 {all_raw_data.shape[1]} 个采样点。")
                self.record_data = []
                self.action_record.setChecked(False)
                self.action_record.setText('开始存盘 (保存完整通道)')
                self.recording = False
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def update_loop(self):
        if not self.server_running: return

        if self.connection_start_time is not None and not self.playback_mode:
            elapsed = int(time.time() - self.connection_start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.time_label.setText(f"已连接: {hours:02d}:{minutes:02d}:{seconds:02d}")
        elif self.playback_mode:
            sec = int(self.playback_ptr / self.sampling_rate)
            self.time_label.setText(f"播放进度: {sec}s")

        if self.playback_mode and self.playback_raw:
            n_win = int(self.sampling_rate * self.window_size)
            start = self.playback_ptr
            stop = start + n_win
            if stop >= self.playback_raw.n_times:
                self.playback_ptr = 0
                return
            data = self.playback_raw.get_data(start=start, stop=stop)
            self.playback_ptr += int(self.sampling_rate * (self.update_speed_ms / 1000.0))
        else:
            if self.dataloader is None: return
            data = self.dataloader.get_latest_data(duration=self.window_size)

        if self.processing_paused or data is None or not np.any(data != 0):
            return

        if self.server_running and not self.playback_mode and not getattr(self, 'online_trigger_mapped', False):
            ch_names = getattr(self.dataloader, 'ch_names', None)
            if not ch_names and hasattr(self.dataloader, 'info') and self.dataloader.info:
                ch_names = self.dataloader.info.get('ch_names')

            if ch_names:
                if 'TRG001' in ch_names:
                    self.trigger_idx_9hz = ch_names.index('TRG001')
                    self.idx9_input.setText(str(self.trigger_idx_9hz))
                if 'TRG002' in ch_names:
                    self.trigger_idx_14hz = ch_names.index('TRG002')
                    self.idx14_input.setText(str(self.trigger_idx_14hz))
                self.online_trigger_mapped = True

        actual_n_channels = data.shape[0]
        if actual_n_channels != self.last_n_channels:
            self._update_dynamic_channels(actual_n_channels)
            if not self.playback_mode:
                self.sampling_rate = self.dataloader.fs
                self.processor.sampling_rate = self.sampling_rate

        if self.recording:
            self.record_data.append(data.copy())

        display_data = data.copy()
        meg_limit = min(306, actual_n_channels)
        display_data[:meg_limit, :] = display_data[:meg_limit, :] * 1e15

        should_predict = False
        ground_truth = "无刺激"
        ui_status_text = "静息态 (等待单一脉冲打标触发...)"

        if self.trigger_idx_14hz < actual_n_channels and self.trigger_idx_9hz < actual_n_channels:
            trg_9_clean, trg_14_clean, should_predict, ground_truth, ui_status_text = self.processor.process_fsm(
                data, self.trigger_idx_9hz, self.trigger_idx_14hz, self.trg_threshold
            )
            self.trigger_curve_9hz.setData(trg_9_clean)
            self.trigger_curve_14hz.setData(trg_14_clean)

        visual_data_filtered = self.processor.filter_display_data(display_data, self.display_indices, self.filter_params)
        for idx, ch_data in enumerate(visual_data_filtered):
            self.curves[idx].setData(ch_data)

        if len(visual_data_filtered) > 0:
            v_data = np.array(visual_data_filtered)
            nperseg = min(int(self.sampling_rate) * 2, v_data.shape[1])
            try:
                freqs, psds = welch(v_data, fs=int(self.sampling_rate), nperseg=nperseg)
                valid_idx = np.where((freqs >= 2) & (freqs <= 30))[0]
                if len(valid_idx) > 0:
                    self.psd_curve.setData(freqs[valid_idx], np.mean(psds, axis=0)[valid_idx])
            except Exception:
                pass

            is_loaded = self.classifier is not None and hasattr(self.classifier,
                                                                'model') and self.classifier.model is not None
            if self.classification_enabled and is_loaded:
                if should_predict:
                    self._perform_classification(data, ground_truth)
                else:
                    self.prediction_label.setText(ui_status_text)
                    if self.processor.trial_state == "COLLECTING":
                        self.prediction_label.setStyleSheet(
                            "font-size: 16pt; font-weight: bold; padding: 10px; border: 2px solid #2196F3; border-radius: 5px; background-color: #E3F2FD;")
                    elif self.processor.trial_state == "RESTING":
                        self.prediction_label.setStyleSheet(
                            "font-size: 16pt; font-weight: bold; padding: 10px; border: 2px solid #FF9800; border-radius: 5px; background-color: #FFF3E0;")
                    else:
                        self.prediction_label.setStyleSheet(
                            "font-size: 16pt; font-weight: bold; padding: 10px; border: 2px solid gray; border-radius: 5px; background-color: #f0f0f0;")

                    self.confidence_label.setText("置信度: --")
                    if "静息态" in ui_status_text or "冷却期" in ui_status_text:
                        self._send_serial_cmd(b'\xEE')

        self.frame_count += 1
        if self.frame_count % 3 == 0 and hasattr(self, 'layout') and self.layout is not None:
            try:
                meg_data = display_data[:306, :]
                f, p = welch(meg_data, fs=int(self.sampling_rate), nperseg=meg_data.shape[1])

                idx_9hz = np.argmin(np.abs(f - 9.0))
                idx_14hz = np.argmin(np.abs(f - 14.0))
                powers = p[:, idx_9hz] + p[:, idx_14hz]

                p_max = np.percentile(powers, 95)
                powers_clipped = np.clip(powers, a_min=0, a_max=p_max)

                n_sensors = min(306, len(self.layout.pos))
                self.topo_canvas.update_topo(powers_clipped[:n_sensors], self.layout.pos)
            except Exception:
                pass

        QtWidgets.QApplication.processEvents()

    def _perform_classification(self, data, ground_truth):
        try:
            prediction, confidence = self.classifier.predict(data, self.sampling_rate)
            freq_text = self.classifier.get_class_name(prediction)
            freq_text = freq_text.replace("8Hz", "9Hz").replace("15Hz", "14Hz")

            display_text = f"真实: {ground_truth} | AI预测: {freq_text}"

            if prediction == 0:
                self._send_serial_cmd(b'\x11')
            else:
                self._send_serial_cmd(b'\x22')

            if ground_truth in ["9Hz", "14Hz"]:
                self.stat_total_trials += 1

                if ground_truth in freq_text:
                    self.stat_correct += 1
                else:
                    self.stat_wrong += 1

                if confidence > 0.6:
                    self.stat_high_confidence += 1

                acc = (self.stat_correct / self.stat_total_trials) * 100 if self.stat_total_trials > 0 else 0.0
                stats_str = f"准确率: {acc:.1f}% ({self.stat_correct}对/{self.stat_wrong}错) | 高置信度(>0.6): {self.stat_high_confidence}次"
                self.stats_label.setText(stats_str)

            if self.classifier.is_confident(confidence, threshold=0.6):
                self.prediction_label.setText(display_text)
                self.confidence_label.setText(f"置信度: {confidence:.2%}")
                color = "background-color: lightblue;" if prediction == 0 else "background-color: lightgreen;"
                self.prediction_label.setStyleSheet(
                    f"font-size: 16pt; font-weight: bold; padding: 10px; border: 2px solid gray; border-radius: 5px; {color}")
            else:
                self.prediction_label.setText(f"{display_text} (不稳定)")
                self.confidence_label.setText(f"置信度: {confidence:.2%}")
                self.prediction_label.setStyleSheet(
                    "font-size: 16pt; font-weight: bold; padding: 10px; border: 2px solid orange; border-radius: 5px; background-color: lightyellow;")
        except Exception:
            pass


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9090)
    parser.add_argument('--com', default='COM6')
    parser.add_argument('--no-dialog', action='store_true')
    args = parser.parse_args()

    app = QtWidgets.QApplication.instance()
    if app is None: app = QtWidgets.QApplication(sys.argv)

    if not args.no_dialog:
        dialog = ConnectionDialog()
        if dialog.exec_() != QDialog.Accepted: return
        params = dialog.get_connection_params()
        host = params['ip']
        port = params['port']
        com = params['com']
    else:
        host = args.host
        port = args.port
        com = args.com

    print(f"启动 MEG-GO UI 框架，准备在 {host}:{port} 进行受控监听")
    try:
        # ⚠️ 修复：默认窗口对齐 4.5 秒
        controller = MEGSystemController(host=host, port=port, com_port=com, window_size=4.5, update_speed_ms=1000)
        controller.show()
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")


if __name__ == '__main__':
    main()
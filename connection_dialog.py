import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QLabel, QDialogButtonBox


class ConnectionDialog(QDialog):
    """
    启动时的网络与硬件连接配置弹窗
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MEG数据源配置")
        self.setModal(True)
        layout = QFormLayout(self)

        self.ip_input = QLineEdit("127.0.0.1")
        layout.addRow("IP地址:", self.ip_input)

        self.port_input = QLineEdit("9090")
        layout.addRow("端口号:", self.port_input)

        self.com_input = QLineEdit("COM5")
        layout.addRow("硬件串口 (无需则留空):", self.com_input)

        info_label = QLabel("提示：当前监听端口已与 C++ 端默认配置(9090)完全对齐")
        info_label.setStyleSheet("color: blue; font-style: italic;")
        layout.addRow(info_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

    def get_connection_params(self):
        return {
            'ip': self.ip_input.text(),
            'port': int(self.port_input.text()),
            'com': self.com_input.text().strip()
        }
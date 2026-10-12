import pyqtgraph as pg

pg.setConfigOption('background', 'w')
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QScrollArea, QAction

from topo_widget import TopoCanvas


class MainWindowUI(QtWidgets.QMainWindow):
    """
    主程序界面的视图构建类。仅负责控件的实例化和布局排版。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle('MEG-GO SSVEP System ')
        self.resize(1650, 950)

    def setup_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        self.title = QLabel("MEG Signal Preview & Real-Time PSD")
        self.title.setStyleSheet("font-size: 20pt; font-weight: bold; margin: 10px; color: black;")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(self.title)

        self._build_classification_display()
        self._build_trigger_plot()
        self._build_psd_plot()
        self._build_topomap_area()
        self._build_timeseries_area()
        self._build_menubar()

    def _build_classification_display(self):
        self.classification_widget = QWidget()
        self.classification_layout = QHBoxLayout(self.classification_widget)

        self.btn_server = QPushButton("启动实时监听")
        self.btn_server.setFixedSize(140, 50)
        self.btn_server.setStyleSheet(
            "font-size: 13pt; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 5px;")

        self.btn_pause = QPushButton("暂停界面")
        self.btn_pause.setFixedSize(140, 50)
        self.btn_pause.setStyleSheet(
            "font-size: 13pt; font-weight: bold; background-color: #FF9800; color: white; border-radius: 5px;")
        self.btn_pause.setEnabled(False)

        self.time_label = QLabel("已连接: --:--:--")
        self.time_label.setStyleSheet(
            "font-size: 13pt; font-weight: bold; color: #333; padding: 10px; border: 2px dashed #ccc; border-radius: 5px;")
        self.time_label.setAlignment(QtCore.Qt.AlignCenter)

        self.prediction_label = QLabel("静息态")
        self.prediction_label.setStyleSheet(
            "font-size: 14pt; font-weight: bold; padding: 10px; border: 2px solid gray; border-radius: 5px; background-color: #f0f0f0;")
        self.prediction_label.setAlignment(QtCore.Qt.AlignCenter)

        self.confidence_label = QLabel("置信度: --")
        self.confidence_label.setStyleSheet("font-size: 13pt; padding: 10px;")

        self.stats_label = QLabel("准确率: 0.0% (0对/0错) | 高置信度(>0.6): 0次")
        self.stats_label.setStyleSheet(
            "font-size: 13pt; font-weight: bold; color: #000; padding: 10px; border: 1px solid #999; border-radius: 5px; background-color: #e8f5e9;")
        self.stats_label.setAlignment(QtCore.Qt.AlignCenter)

        # 初始状态占位，控制器后续会更新
        self.model_status_label = QLabel("正在初始化...")

        self.classification_layout.addWidget(self.btn_server)
        self.classification_layout.addWidget(self.btn_pause)
        self.classification_layout.addWidget(self.time_label)
        self.classification_layout.addWidget(self.prediction_label)
        self.classification_layout.addWidget(self.confidence_label)
        self.classification_layout.addWidget(self.stats_label)
        self.classification_layout.addWidget(self.model_status_label)

        self.main_layout.addWidget(self.classification_widget)

    def _build_trigger_plot(self):
        self.trigger_container = QWidget()
        self.trigger_layout = QVBoxLayout(self.trigger_container)
        self.trigger_layout.setContentsMargins(0, 0, 0, 0)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("[打标通道在线校准]"))

        ctrl_layout.addWidget(QLabel("9Hz 索引:"))
        self.idx9_input = QLineEdit(str(getattr(self, 'trigger_idx_9hz', 320)))
        self.idx9_input.setFixedWidth(50)
        ctrl_layout.addWidget(self.idx9_input)

        ctrl_layout.addWidget(QLabel("14Hz 索引:"))
        self.idx14_input = QLineEdit(str(getattr(self, 'trigger_idx_14hz', 321)))
        self.idx14_input.setFixedWidth(50)
        ctrl_layout.addWidget(self.idx14_input)

        ctrl_layout.addWidget(QLabel("绝对阈值:"))
        self.thresh_input = QLineEdit(str(getattr(self, 'trg_threshold', 1.0)))
        self.thresh_input.setFixedWidth(50)
        ctrl_layout.addWidget(self.thresh_input)

        self.btn_apply_trg = QPushButton("应用设置")
        ctrl_layout.addWidget(self.btn_apply_trg)

        self.btn_clear_stats = QPushButton("重置统计")
        ctrl_layout.addWidget(self.btn_clear_stats)
        ctrl_layout.addStretch()

        self.trigger_layout.addLayout(ctrl_layout)

        self.trigger_plot = pg.PlotWidget(title="刺激打标双通道直连监控 (红线=9Hz / 绿线=14Hz)")
        self.trigger_plot.setFixedHeight(120)
        self.trigger_plot.setLabel('left', 'TTL 电平')
        self.trigger_plot.addLegend(offset=(10, 10))

        self.trigger_curve_9hz = self.trigger_plot.plot(pen=pg.mkPen('r', width=3), name="9Hz")
        self.trigger_curve_14hz = self.trigger_plot.plot(pen=pg.mkPen('g', width=3), name="14Hz")

        self.trigger_layout.addWidget(self.trigger_plot)
        self.main_layout.addWidget(self.trigger_container)

    def _build_psd_plot(self):
        self.psd_plot = pg.PlotWidget(title="实时频谱图 (PSD) - 视觉通道平均能量")
        self.psd_plot.setFixedHeight(220)
        self.psd_plot.setLabel('left', 'Power', units='fT^2/Hz')
        self.psd_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.psd_plot.setXRange(2, 30)
        self.psd_plot.showGrid(x=True, y=True, alpha=0.3)

        vline9 = pg.InfiniteLine(angle=90, movable=False, pos=9, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
        vline14 = pg.InfiniteLine(angle=90, movable=False, pos=14, pen=pg.mkPen('g', width=2, style=QtCore.Qt.DashLine))
        self.psd_plot.addItem(vline9)
        self.psd_plot.addItem(vline14)

        self.psd_curve = self.psd_plot.plot(pen=pg.mkPen('k', width=2))
        self.main_layout.addWidget(self.psd_plot)

    def _build_topomap_area(self):
        self.topo_container = QWidget()
        self.topo_layout = QHBoxLayout(self.topo_container)

        self.topo_label = QLabel("实时 Topography\n(平滑能量定位)")
        self.topo_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        self.topo_label.setFixedWidth(120)

        self.topo_canvas = TopoCanvas(self.topo_container)
        self.topo_layout.addWidget(self.topo_label)
        self.topo_layout.addWidget(self.topo_canvas, stretch=1)
        self.topo_container.setFixedHeight(280)
        self.main_layout.addWidget(self.topo_container)

    def _build_timeseries_area(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

        self.plots = []
        self.curves = []
        self.legends = []
        for i in range(24):
            pw = pg.PlotWidget()
            pw.setFixedHeight(90)
            pw.hideAxis('left')
            pw.hideAxis('bottom')
            curve = pw.plot(pen='b')
            legend = pg.LegendItem(offset=(0, 0))
            legend.setParentItem(pw.getPlotItem())
            # 名称后续由控制器更新
            legend.addItem(curve, f"CH{i + 1}")
            self.scroll_layout.addWidget(pw)
            self.plots.append(pw)
            self.curves.append(curve)
            self.legends.append(legend)
        self.scroll_layout.addStretch(1)

    def _build_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu('文件(重载与保存)')
        self.action_load_fif = QAction('加载离线 FIF 文件 (数据重演)', self)
        file_menu.addAction(self.action_load_fif)
        self.action_save_fif = QAction('导出已收数据 (结果重载)', self)
        file_menu.addAction(self.action_save_fif)

        filter_menu = menubar.addMenu('滤波')
        self.action_bandpass = QAction('带通滤波', self, checkable=True)
        self.action_bandpass.setChecked(True)
        filter_menu.addAction(self.action_bandpass)
        self.action_notch = QAction('陷波滤波', self, checkable=True)
        filter_menu.addAction(self.action_notch)

        classification_menu = menubar.addMenu('分类')
        self.action_enable_class = QAction('启用AI分类', self, checkable=True)
        self.action_enable_class.setChecked(True)
        classification_menu.addAction(self.action_enable_class)

        record_menu = menubar.addMenu('记录缓冲区')
        self.action_record = QAction('开始存盘 (保存完整通道)', self, checkable=True)
        record_menu.addAction(self.action_record)
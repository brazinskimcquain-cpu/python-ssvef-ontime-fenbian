import numpy as np
import mne
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.collections
from matplotlib.ticker import ScalarFormatter
from PyQt5.QtWidgets import QWidget, QVBoxLayout


class TopoCanvas(QWidget):
    """
    独立封装的 Matplotlib 脑磁地形图渲染组件
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')

        self.colorbar = None
        self.layout.addWidget(self.canvas)

    def update_topo(self, plot_data, pos_layout):
        # 防崩溃垫片
        vmin, vmax = np.min(plot_data), np.max(plot_data)
        if vmax - vmin < 1e-12:
            vmax += 1e-12

        # 提取有效传感器的坐标并进行中心归一化
        n_sensors = len(plot_data)
        plot_pos = pos_layout[:n_sensors, :2].copy()
        plot_pos[:, 0] -= np.mean(plot_pos[:, 0])
        plot_pos[:, 1] -= np.mean(plot_pos[:, 1])

        # 缩放到 0.5 半径匹配标准头部轮廓
        max_radius = np.max(np.linalg.norm(plot_pos, axis=1))
        if max_radius > 0:
            plot_pos = (plot_pos / max_radius) * 0.5

        # 彻底清空并重建图层以防 Colorbar 侵占空间
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        try:
            returns = mne.viz.plot_topomap(
                plot_data,
                plot_pos,
                axes=self.ax,
                show=False,
                cmap='Reds',
                extrapolate='local',
                image_interp='cubic',
                res=256,
                contours=5,
                vlim=(vmin, vmax),
                sphere=(0.0, 0.0, 0.0, 0.5),
                sensors=True
            )

            im = returns[0] if isinstance(returns, tuple) else returns
            cn = returns[1] if isinstance(returns, tuple) and len(returns) > 1 else None

            # 图像二次平滑
            if hasattr(im, 'set_interpolation'):
                im.set_interpolation('bicubic')

            # 细腻化半透明等高线
            if cn is not None and hasattr(cn, 'collections'):
                for c in cn.collections:
                    c.set_linewidth(0.5)
                    c.set_alpha(0.3)
                    if hasattr(c, 'set_color'):
                        c.set_color('#333333')

            # 极小化黑色传感器点
            for coll in self.ax.collections:
                if isinstance(coll, matplotlib.collections.PathCollection):
                    coll.set_facecolor('k')
                    coll.set_edgecolor('none')
                    coll.set_alpha(0.8)
                    coll.set_sizes([1.5])
                    break

            # 科学计数法格式化颜色条
            if im:
                fmt = ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((-1, 3))

                self.colorbar = self.fig.colorbar(
                    im, ax=self.ax, orientation='horizontal',
                    fraction=0.05, pad=0.05, shrink=0.7, aspect=40,
                    format=fmt
                )
                self.colorbar.set_label('Power (fT^2/Hz)', fontsize=8)
                self.colorbar.ax.tick_params(labelsize=7)
                self.colorbar.ax.xaxis.get_offset_text().set_fontsize(7)

            self.ax.set_aspect('equal')
            self.ax.axis('off')
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"[!] Topomap 渲染错误: {e}")
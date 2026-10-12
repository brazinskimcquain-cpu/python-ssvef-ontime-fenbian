import os
import gc
import mne
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from scipy.linalg import svd
from datetime import datetime


def apply_mnf_denoising(raw, n_components_to_drop=2):
    """MNF/PCA 动态空间滤波器"""
    print(f"[*] 启动动态空间降噪引擎 (MNF/PCA 模式)，剔除前 {n_components_to_drop} 个全局环境噪声主成分...")
    raw_clean = raw.copy()
    picks = mne.pick_types(raw_clean.info, meg='grad', exclude='bads')
    if len(picks) == 0:
        return raw_clean

    data = raw_clean.get_data(picks=picks)
    mean_data = np.mean(data, axis=1, keepdims=True)
    data_centered = data - mean_data

    # SVD 对中心化数据分解，数学上等价于 PCA
    U, S, V = svd(data_centered, full_matrices=False)
    S_clean = S.copy()
    S_clean[:n_components_to_drop] = 0

    data_reconstructed = U @ np.diag(S_clean) @ V + mean_data
    raw_clean._data[picks] = data_reconstructed
    print("[+] MNF/PCA 空间降噪完成！")
    return raw_clean


def automated_cleaning_pipeline():
    print("=" * 70)
    print("🧠 欢迎使用 MEG 极速预处理与清洗导出引擎 (纯净导出版)")
    print("=" * 70)

    # 根据当前精确时间创建全局输出文件夹
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f'CLEANED_FIF_{current_time_str}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] 清洗后数据输出目录已就绪: {output_dir}")

    # GUI 收集独立文件夹
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("[*] 正在呼出文件夹选择窗口...")
    target_dirs = []
    while True:
        title_text = f"已添加 {len(target_dirs)} 个文件夹。请选择下一个，或点【取消】开始运行"
        d = filedialog.askdirectory(title=title_text, parent=root)
        if not d:
            break
        if d not in target_dirs:
            target_dirs.append(d)
            print(f"  [+] 成功添加待处理文件夹: {d}")
    root.destroy()

    if not target_dirs:
        print("[!] 未选择任何文件夹，程序退出。")
        return

    print(f"\n[*] 开始在选中的范围内展开地毯式搜索...")
    fif_file_paths = []
    for target_dir in target_dirs:
        for root_path, dirs, files in os.walk(target_dir):
            for file in files:
                # 避开已经处理过的文件或 epoch/ica 衍生文件
                if file.endswith('.fif') and not file.endswith('-epo.fif') and not file.endswith(
                        '-ica.fif') and 'clean' not in file:
                    fif_file_paths.append(os.path.join(root_path, file))

    total_files = len(fif_file_paths)
    if total_files == 0:
        print(f"[!] 未找到任何原始 .fif 文件！程序退出。")
        return
    print(f"\n[🎉 扫荡成功] 共揪出 {total_files} 个 .fif 文件！")

    for idx, current_file in enumerate(fif_file_paths, start=1):
        base_name = Path(current_file).stem
        print("\n" + "🚀" * 30)
        print(f"▶️ [处理进度: {idx} / {total_files}] 当前处理: 【{base_name}】")

        try:
            # 1. 载入原始数据
            raw = mne.io.read_raw_fif(current_file, preload=True, verbose=False)

            # 2. 物理剔除标记为 bads 的坏通道，保留剩下所有健康的通道
            if raw.info['bads']:
                print(f"[*] 剔除坏通道: {raw.info['bads']}")
                raw.drop_channels(raw.info['bads'])
            else:
                print("[*] 没有发现标记为 bads 的坏通道，保留所有当前通道。")

            # 3. 空间降噪
            raw = apply_mnf_denoising(raw)

            # 4. 带通滤波
            print("[*] 进行全局带通滤波 (3.0 - 45.0 Hz)...")
            raw.filter(l_freq=3.0, h_freq=45.0, fir_design='firwin', verbose=False)

            # 5. 20Hz & 40Hz 陷波手术刀
            print("[*] 🔪 启动手术刀：执行 20Hz 及 40Hz 陷波滤波 (Notch Filter)...")
            raw.notch_filter(freqs=[20.0, 40.0], fir_design='firwin', verbose=False)

            # 6. 全局连续数据 Z-score 归一化 (通道内计算)
            print("[*] 执行通道内 Z-score 归一化...")
            data = raw.get_data()  # 获取全量数据矩阵 [channels, times]
            mean_val = np.mean(data, axis=1, keepdims=True)
            std_val = np.std(data, axis=1, keepdims=True)

            # 防止数据方差为0导致的除以0错误 (除以1等价于不进行缩放)
            std_val[std_val == 0] = 1.0

            data_normalized = (data - mean_val) / std_val
            raw._data = data_normalized

            # 7. 另存为新的 FIF 文件
            save_name = f"{base_name}_clean_raw.fif"
            save_path = os.path.join(output_dir, save_name)
            print(f"[*] 正在保存清洗后的连续数据至: {save_name}")

            # fmt="single" 可以稍微压缩文件体积, overwrite=True 允许覆盖
            raw.save(save_path, overwrite=True, fmt="single", verbose=False)
            print("[+] 保存成功！")

        except Exception as e:
            print(f"\n[❌ 致命错误] 文件 【{base_name}】 处理失败: {e}")
            continue

        finally:
            # 严格的内存管理
            if 'raw' in locals(): del raw
            if 'data' in locals(): del data
            if 'data_normalized' in locals(): del data_normalized
            gc.collect()

    print("\n" + "=" * 70)
    print("🏆 终极捷报！全部数据清洗并保存完毕！")
    print(f"一共处理了 {total_files} 个文件。")
    print(f"所有经过滤波、归一化、只含健康通道的 FIF 文件已集中存放于:\n👉 {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    automated_cleaning_pipeline()
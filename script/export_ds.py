"""
从 ADS 数据集 (.ds) 读取数据并导出为 pickle 二进制文件（dict）。

需使用 ADS 自带的 Python 环境运行（只有该环境提供 keysight.ads.dataset）：

    "C:/Program Files/Keysight/ADS2026/tools/python/python.exe" script/export_ds.py "C:/Users/YangHui/Documents/ADS/MagnetWirelessCommunication/data/tx_pa.ds"

导出后在普通 Jupyter 中用 pickle.load 即可恢复 data 字典，无需 keysight 环境。
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


def export_ds(dat_file_path: str | Path, out_dir: str | Path = "out") -> Path:
    """读取 ADS .ds 数据集，构建 data 字典，并序列化为二进制文件。

    Parameters
    ----------
    dat_file_path : str | Path
        ADS 数据集 (.ds) 的路径。
    out_dir : str | Path
        输出目录，默认为当前目录下的 out/。

    Returns
    -------
    Path
        导出的二进制文件路径（.pkl）。
    """
    import keysight.ads.dataset as dataset  # 仅 ADS Python 环境可用，故延迟导入

    dat_file_path = Path(dat_file_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_data = dataset.open(str(dat_file_path))
    data: dict[str, dict[Any, Any]] = {}

    for varblock in output_data.varblock_names:
        data[varblock] = {}
        dataframe = output_data[varblock].to_dataframe().reset_index()
        for column_index in range(len(dataframe.columns)):
            column_name = dataframe.columns[column_index]
            data[varblock][column_name] = dataframe.values[:, column_index]

    out_path = out_dir / (dat_file_path.stem + ".pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    for name, cols in data.items():
        print(f"  - {name}: {list(cols.keys())}")
    return out_path


def load_data(pkl_path: str | Path) -> dict:
    """在 Jupyter 中恢复 data 字典（仅依赖标准库，无需 keysight）。"""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 ADS .ds 数据集为 pickle 二进制文件")
    parser.add_argument("dat_file_path", help="ADS 数据集 (.ds) 的路径")
    parser.add_argument("--out-dir", default="out", help="输出目录（默认 out/）")
    args = parser.parse_args(argv)

    export_ds(args.dat_file_path, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

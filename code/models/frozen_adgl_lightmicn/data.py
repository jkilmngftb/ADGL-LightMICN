"""Leakage-safe loading and chronological splitting of multistation temperature data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import hashlib

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    """All arrays and metadata needed by the model and audit code."""

    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    x_mid_train: Optional[np.ndarray]
    x_mid_val: Optional[np.ndarray]
    x_mid_test: Optional[np.ndarray]
    x_long_train: Optional[np.ndarray]
    x_long_val: Optional[np.ndarray]
    x_long_test: Optional[np.ndarray]
    x_lag_train: Optional[np.ndarray]
    x_lag_val: Optional[np.ndarray]
    x_lag_test: Optional[np.ndarray]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_raw_train: np.ndarray
    y_raw_val: np.ndarray
    y_raw_test: np.ndarray
    calendar_train: np.ndarray
    calendar_val: np.ndarray
    calendar_test: np.ndarray
    date_train: np.ndarray
    date_val: np.ndarray
    date_test: np.ndarray
    station_ids: tuple
    adj: np.ndarray
    scaler_min: np.ndarray
    scaler_max: np.ndarray
    split_boundaries: dict
    data_path: str
    data_sha256: str
    feature_names: tuple = ()
    sample_weight_train: Optional[np.ndarray] = None
    sample_weight_val: Optional[np.ndarray] = None
    sample_weight_test: Optional[np.ndarray] = None
    feature_min: Optional[np.ndarray] = None
    feature_range: Optional[np.ndarray] = None
    protocol: str = "legacy"

    @property
    def has_multiscale(self) -> bool:
        return self.x_mid_train is not None and self.x_long_train is not None


    @property
    def num_nodes(self) -> int:
        return int(self.x_train.shape[2])

    @property
    def seq_len(self) -> int:
        return int(self.x_train.shape[1])

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_adjacency(
    adj_path: Path,
    station_ids: Sequence,
    adj_station_ids: Optional[Sequence] = None,
    allow_positional_adj: bool = False,
) -> np.ndarray:
    adj = np.loadtxt(adj_path, delimiter=",", dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"邻接矩阵必须是方阵，得到 {adj.shape}")
    if adj.shape[0] != len(station_ids):
        raise ValueError(f"邻接矩阵节点数 {adj.shape[0]} 与站点数 {len(station_ids)} 不一致")
    if adj_station_ids is None and not allow_positional_adj:
        raise ValueError(
            "adjacency_matrix.csv 没有站点 ID 映射；请提供 adj_station_ids，或显式设置 "
            "allow_positional_adj=True 以确认按数据列顺序对齐"
        )
    if adj_station_ids is not None and list(adj_station_ids) != list(station_ids):
        raise ValueError("邻接矩阵站点顺序与温度数据站点顺序不一致")
    if not np.isfinite(adj).all():
        raise ValueError("邻接矩阵包含非有限值")
    if np.any(adj < 0):
        raise ValueError("邻接矩阵不能包含负边权")
    return adj.astype(np.float32)


def periodic_features(dates: Sequence, seq_len: int) -> np.ndarray:
    """Return annual, semi-annual, and within-window sinusoidal encodings."""
    dates = pd.to_datetime(np.asarray(dates))
    if len(dates) != seq_len:
        raise ValueError(f"dates 长度 {len(dates)} 与 seq_len={seq_len} 不一致")
    day_of_year = dates.dayofyear.to_numpy(dtype=np.float32)
    year_length = 365.2425
    annual = 2.0 * np.pi * day_of_year / year_length
    relative = 2.0 * np.pi * np.arange(seq_len, dtype=np.float32) / float(seq_len)
    relative = np.broadcast_to(relative[None, :], (len(dates), seq_len))
    annual = np.broadcast_to(annual[:, None], (len(dates), seq_len))
    return np.stack(
        [
            np.sin(annual),
            np.cos(annual),
            np.sin(2.0 * annual),
            np.cos(2.0 * annual),
            np.sin(relative),
            np.cos(relative),
        ],
        axis=-1,
    ).astype(np.float32)


def _scale(values: np.ndarray, scaler_min: np.ndarray, scaler_max: np.ndarray) -> np.ndarray:
    denom = np.where((scaler_max - scaler_min) < 1e-8, 1.0, scaler_max - scaler_min)
    return ((values - scaler_min) / denom).astype(np.float32)


def inverse_scale(values: np.ndarray, scaler_min: np.ndarray, scaler_max: np.ndarray) -> np.ndarray:
    denom = np.where((scaler_max - scaler_min) < 1e-8, 1.0, scaler_max - scaler_min)
    return np.asarray(values) * denom + scaler_min


COMMON_FEATURE_COLS = (
    "平均气温", "最高气温", "最低气温", "平均风速",
    "平均相对湿度", "降水量", "平均气压", "日照时数",
)


def _read_common_table(path: Path, feature_names: Sequence[str]) -> tuple[pd.DataFrame, tuple]:
    """Read the same header/unit-row convention as common.py."""
    raw = pd.read_excel(path, header=0)
    if len(raw) < 2:
        raise ValueError("Excel 数据不足以删除单位行")
    raw = raw.iloc[1:].reset_index(drop=True)
    raw = raw.rename(columns={"站号": "sid", "年": "year", "月": "month", "日": "day"})
    required = {"sid", "year", "month", "day", *feature_names}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Excel 缺少 common.py 所需字段: {sorted(missing)}")
    for col in ["year", "month", "day", *feature_names]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if raw[["sid", "year", "month", "day", *feature_names]].isna().any().any():
        raise ValueError("common.py 协议要求站号、日期和 8 个特征均无缺失")
    raw["date"] = pd.to_datetime(dict(year=raw.year, month=raw.month, day=raw.day), errors="coerce")
    if raw.date.isna().any():
        raise ValueError("存在无法解析的日期")
    sids = tuple(sorted(raw.sid.unique().tolist()))
    return raw, sids


def _common_extreme_weights(temp_raw, months, train_end, val_end, seq_len,
                            threshold=1.5, weight=3.0, cap=5.0):
    """Reproduce common.py's train-only monthly anomaly weights."""
    temp_raw = np.asarray(temp_raw, dtype=np.float32)
    months = np.asarray(months, dtype=np.int32)
    _, nodes = temp_raw.shape
    mu = np.zeros((nodes, 13), np.float32)
    sd = np.ones((nodes, 13), np.float32)
    for month in range(1, 13):
        sel = months[:train_end] == month
        if not np.any(sel):
            continue
        mu[:, month] = temp_raw[:train_end][sel].mean(axis=0)
        std = temp_raw[:train_end][sel].std(axis=0)
        sd[:, month] = np.where(std < 1e-6, 1.0, std)

    def build(start, end):
        count = max(0, end - start - seq_len)
        if count == 0:
            return np.empty((0, nodes), np.float32)
        target_idx = start + np.arange(seq_len, seq_len + count)
        vals = temp_raw[target_idx]
        mm = months[target_idx]
        mu_s = mu.T[mm]
        sd_s = sd.T[mm]
        z = np.abs((vals - mu_s) / sd_s)
        excess = np.maximum(z - threshold, 0.0)
        return np.minimum(1.0 + weight * excess, cap).astype(np.float32)

    return build(0, train_end), build(train_end, val_end), build(val_end, len(temp_raw))


def load_common_xlsx(
    data_xlsx: str | Path,
    adj_path: str | Path,
    seq_len: int = 24,
    train_rate: float = 0.8,
    val_ratio: float = 0.10,
    feature_names: Sequence[str] = COMMON_FEATURE_COLS,
    adj_station_ids: Optional[Sequence] = None,
    allow_positional_adj: bool = False,
) -> DataBundle:
    """Load the exact data protocol implemented by ``common.py``.

    Windows are made separately inside the 70/10/20 chronological segments;
    statistics are fitted only on the pure 70% training segment.
    """
    if seq_len < 1:
        raise ValueError("seq_len 必须为正数")
    data_path = Path(data_xlsx).expanduser().resolve()
    adj_file = Path(adj_path).expanduser().resolve()
    raw, station_ids = _read_common_table(data_path, feature_names)
    frames = []
    for col in feature_names:
        pivot = raw.pivot_table(index="date", columns="sid", values=col).sort_index()
        pivot = pivot.reindex(columns=station_ids)
        frames.append(pivot.to_numpy(dtype=np.float32))
    data_raw = np.stack(frames, axis=-1)  # [T,N,F]
    dates = raw[["date"]].drop_duplicates().sort_values("date")["date"].to_numpy()
    if data_raw.shape[0] != len(dates) or not np.isfinite(data_raw).all():
        raise ValueError("特征透视后存在日期不齐、缺失或非有限值")
    if len(dates) < 3 * seq_len + 3:
        raise ValueError("数据长度不足以在三段中构造窗口")
    if not np.all(np.diff(dates).astype("timedelta64[D]") == np.timedelta64(1, "D")):
        raise ValueError("日期必须是连续日频时间轴")

    n_days, nodes, n_feat = data_raw.shape
    train_end = int(n_days * (train_rate - val_ratio))
    val_end = int(n_days * train_rate)
    if not (0 < train_end < val_end < n_days):
        raise ValueError("train_rate/val_ratio 导致无效切分")
    scaled = np.empty_like(data_raw, dtype=np.float32)
    feat_min = np.zeros(n_feat, np.float32)
    feat_range = np.ones(n_feat, np.float32)
    for j, name in enumerate(feature_names):
        col = data_raw[:, :, j]
        if name == "降水量":
            transformed = np.log1p(np.clip(col, 0.0, None))
            mu = float(transformed[:train_end].mean())
            sd = float(transformed[:train_end].std())
            sd = sd if sd > 1e-8 else 1.0
            scaled[:, :, j] = np.clip((transformed - mu) / sd, -3.0, 3.0)
            feat_min[j], feat_range[j] = mu, sd
        else:
            mn = float(col[:train_end].min())
            mx = float(col[:train_end].max())
            rng = mx - mn if mx - mn > 1e-8 else 1.0
            scaled[:, :, j] = (col - mn) / rng
            feat_min[j], feat_range[j] = mn, rng

    months = pd.to_datetime(dates).month.to_numpy(dtype=np.int32)
    target_idx = int(list(feature_names).index("平均气温"))

    def make_segment(start, end):
        count = max(0, end - start - seq_len)
        xs, ys, yraw, cals, target_dates = [], [], [], [], []
        for i in range(count):
            abs_start = start + i
            target = abs_start + seq_len
            xs.append(scaled[abs_start:target])
            ys.append(scaled[target, :, target_idx])
            yraw.append(data_raw[target, :, target_idx])
            cals.append(periodic_features(dates[abs_start:target], seq_len)[0])
            target_dates.append(dates[target])
        return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
                np.asarray(yraw, np.float32), np.asarray(cals, np.float32),
                np.asarray(target_dates))

    train = make_segment(0, train_end)
    val = make_segment(train_end, val_end)
    test = make_segment(val_end, n_days)
    weights = _common_extreme_weights(data_raw[:, :, target_idx], months, train_end, val_end, seq_len)
    adj = _read_adjacency(adj_file, station_ids, adj_station_ids, allow_positional_adj)
    split_boundaries = {
        "n_days": n_days,
        "train_end_date_exclusive": str(dates[train_end]),
        "val_end_date_exclusive": str(dates[val_end]),
        "train_targets": int(len(train[0])),
        "val_targets": int(len(val[0])),
        "test_targets": int(len(test[0])),
        "train_rate": train_rate,
        "val_ratio": val_ratio,
        "protocol": "common.py/tr70v10t20_valsel",
    }
    return DataBundle(
        x_train=train[0], x_val=val[0], x_test=test[0],
        x_mid_train=None, x_mid_val=None, x_mid_test=None,
        x_long_train=None, x_long_val=None, x_long_test=None,
        x_lag_train=None, x_lag_val=None, x_lag_test=None,
        y_train=train[1], y_val=val[1], y_test=test[1],
        y_raw_train=train[2], y_raw_val=val[2], y_raw_test=test[2],
        calendar_train=train[3], calendar_val=val[3], calendar_test=test[3],
        date_train=train[4], date_val=val[4], date_test=test[4],
        station_ids=station_ids, adj=adj,
        scaler_min=np.full(nodes, feat_min[target_idx], np.float32),
        scaler_max=np.full(nodes, feat_min[target_idx] + feat_range[target_idx], np.float32),
        split_boundaries=split_boundaries, data_path=str(data_path), data_sha256=_sha256(data_path),
        feature_names=tuple(feature_names), sample_weight_train=weights[0],
        sample_weight_val=weights[1], sample_weight_test=weights[2],
        feature_min=feat_min, feature_range=feat_range, protocol="common.py",
    )

"""指标趋势异常监测：区分一次尖峰（spike）与持续偏移（shift）的纯函数判定。

判定口径（随规则版本固化进告警快照，保证历史判定可复算）：
- 序列前 ``window`` 个非缺失点为基线参考期，本身不参与判定；
- 基线中心为均值，离散度为样本标准差；离散度为 0 时，任何不等于
  中心的点都视为偏离（因此纯重复数据自身永不告警）；
- 偏离分数 ``score = |x - center| / sd``，达到 ``z_threshold`` 记为偏离点；
- 连续偏离点归并为一个事件：连续长度达到 ``min_run`` 判为持续偏移，
  否则判为一次尖峰；
- 缺失点（None）不参与判定，并打断连续性；
- 空序列或不足以形成基线的序列不产生任何告警（杜绝误报）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import fmean, stdev
from typing import Iterator, Sequence


class AnomalyKind(str, Enum):
    """异常类别。"""

    SPIKE = "spike"  # 一次尖峰：未达到持续阈值的短暂偏离
    SHIFT = "shift"  # 持续偏移：连续偏离长度达到 min_run


@dataclass(frozen=True)
class TrendRuleSpec:
    """判定参数；随规则版本固化进告警快照。"""

    window: int  # 基线参考期长度（非缺失点数），>= 2
    z_threshold: float  # 偏离分数阈值，> 0
    min_run: int  # 持续偏移所需的最短连续偏离长度，>= 2


@dataclass(frozen=True)
class Anomaly:
    """一次异常事件。"""

    kind: AnomalyKind
    index: int  # 事件起始点在原始序列中的下标
    length: int  # 连续偏离点数（尖峰通常为 1）
    peak_value: float  # 偏离最大的点的取值
    peak_score: float  # 最大偏离分数
    baseline: float  # 判定时使用的基线中心


def validate_spec(spec: TrendRuleSpec) -> None:
    """参数不合法时抛 ValueError（服务层映射为 ValidationError）。"""
    if not isinstance(spec.window, int) or isinstance(spec.window, bool) \
            or spec.window < 2:
        raise ValueError("window 必须为 >= 2 的整数")
    if not isinstance(spec.z_threshold, (int, float)) \
            or isinstance(spec.z_threshold, bool) \
            or not math.isfinite(spec.z_threshold) or spec.z_threshold <= 0:
        raise ValueError("z_threshold 必须为正的有限数")
    if not isinstance(spec.min_run, int) or isinstance(spec.min_run, bool) \
            or spec.min_run < 2:
        raise ValueError("min_run 必须为 >= 2 的整数（否则尖峰会被误判为偏移）")


def detect_anomalies(series: Sequence[float | None],
                     spec: TrendRuleSpec) -> list[Anomaly]:
    """对序列执行趋势异常判定；空序列、纯重复数据均不产生告警。"""
    validate_spec(spec)
    baseline: list[float] = []
    center = 0.0
    sd = 0.0
    flagged: list[tuple[int, float, float]] = []  # (下标, 取值, 偏离分数)
    for i, x in enumerate(series):
        if x is None:
            continue
        value = float(x)
        if len(baseline) < spec.window:
            baseline.append(value)
            if len(baseline) == spec.window:
                center = fmean(baseline)
                sd = stdev(baseline)
            continue
        score = _score(value, center, sd)
        if score >= spec.z_threshold:
            flagged.append((i, value, score))
    anomalies: list[Anomaly] = []
    for run in _runs(flagged):
        kind = AnomalyKind.SHIFT if len(run) >= spec.min_run else AnomalyKind.SPIKE
        peak = max(run, key=lambda p: p[2])
        anomalies.append(
            Anomaly(kind, run[0][0], len(run), peak[1], peak[2], center)
        )
    return anomalies


def _score(value: float, center: float, sd: float) -> float:
    """偏离分数；零离散基线下，任何变化都视为完全偏离。"""
    if sd == 0.0:
        return 0.0 if value == center else math.inf
    return abs(value - center) / sd


def _runs(flagged: list[tuple[int, float, float]]
          ) -> Iterator[list[tuple[int, float, float]]]:
    """把下标连续的偏离点归并为事件。"""
    current: list[tuple[int, float, float]] = []
    for point in flagged:
        if current and point[0] != current[-1][0] + 1:
            yield current
            current = []
        current.append(point)
    if current:
        yield current

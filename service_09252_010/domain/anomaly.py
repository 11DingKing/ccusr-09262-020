"""指标趋势异常监测：区分一次尖峰（spike）与持续偏移（shift）。

算法（稳健基线法，纯函数、无 IO）：
- 前 ``baseline_window`` 个点用于建立基线：中位数 ± MAD×1.4826（稳健标准差）；
- 基线之后逐点计算稳健残差 ``r = (x - 中位数) / scale``，
  ``|r| >= sigma_threshold`` 记为异常点；
- 连续异常点构成运行段：段长 ``>= min_run`` 判为**持续偏移**，
  否则判为**一次尖峰**；
- 空序列不产生任何告警（杜绝误报）；序列长度不超过基线窗口时同样不判定；
- 缺失值（None）不算异常，并打断运行段（缺口两侧不构成持续）；
- 基线完全平坦（MAD 为 0，如重复数据）时以极小量兜底：
  相同值不报警，任何可见偏离都会触发。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Sequence

# 平坦基线的兜底尺度：足够小，使任何非零偏离都显著；又避免除零。
_FLAT_SCALE = 1e-9

# 建立基线所需的最少有效点数。
_MIN_BASELINE_POINTS = 2


class AnomalyKind(str, Enum):
    """异常类别：一次尖峰 或 持续偏移。"""

    SPIKE = "spike"  # 一次尖峰：短促偏离后恢复
    SHIFT = "shift"  # 持续偏移：连续 min_run 个点以上偏离


class AnomalyVerdict(str, Enum):
    """一次判定的总体结论。"""

    NORMAL = "normal"  # 无异常
    SPIKE = "spike"  # 仅含尖峰
    SHIFT = "shift"  # 含持续偏移（最严重，优先于尖峰）


@dataclass(frozen=True)
class AnomalyRuleSpec:
    """判定参数。服务层由规则版本装配，告警固化其快照作为判定依据。"""

    baseline_window: int = 5  # 基线窗口：前 N 个点建立基线，不参与判定
    sigma_threshold: float = 3.0  # 稳健残差阈值（个 σ）
    min_run: int = 3  # 持续偏移所需的最短连续异常点数


@dataclass(frozen=True)
class Anomaly:
    """一段异常的判定结果，下标相对原始序列。"""

    kind: AnomalyKind
    start_index: int
    end_index: int
    peak_index: int  # 残差绝对值最大的点
    max_residual: float  # 峰值点的带符号稳健残差

    @property
    def direction(self) -> str:
        return "up" if self.max_residual > 0 else "down"


@dataclass(frozen=True)
class AnomalyJudgment:
    """对一条序列的完整判定。"""

    anomalies: tuple[Anomaly, ...]

    @property
    def verdict(self) -> AnomalyVerdict:
        if not self.anomalies:
            return AnomalyVerdict.NORMAL
        if any(a.kind is AnomalyKind.SHIFT for a in self.anomalies):
            return AnomalyVerdict.SHIFT
        return AnomalyVerdict.SPIKE


def detect_anomalies(
    values: Sequence[float | None], spec: AnomalyRuleSpec
) -> AnomalyJudgment:
    """对原始序列执行趋势异常判定。空序列与短序列直接判正常。"""
    points = list(values)
    if len(points) <= spec.baseline_window:
        return AnomalyJudgment(())
    baseline = [v for v in points[: spec.baseline_window] if v is not None]
    if len(baseline) < _MIN_BASELINE_POINTS:
        return AnomalyJudgment(())
    center = median(baseline)
    scale = 1.4826 * median(abs(v - center) for v in baseline)
    if scale <= 0:
        scale = _FLAT_SCALE
    # 与原始序列下标对齐的残差序列；None 表示该点不异常（缺失或未越限）。
    residuals: list[float | None] = []
    for value in points[spec.baseline_window:]:
        if value is None:
            residuals.append(None)
            continue
        residual = (value - center) / scale
        residuals.append(residual if abs(residual) >= spec.sigma_threshold else None)
    return AnomalyJudgment(
        tuple(_collect_runs(residuals, offset=spec.baseline_window,
                            min_run=spec.min_run))
    )


def _collect_runs(
    residuals: list[float | None], *, offset: int, min_run: int
) -> list[Anomaly]:
    """把连续异常点聚成运行段，并按段长分类为尖峰或偏移。"""
    anomalies: list[Anomaly] = []
    i, n = 0, len(residuals)
    while i < n:
        if residuals[i] is None:
            i += 1
            continue
        j = i
        while j + 1 < n and residuals[j + 1] is not None:
            j += 1
        peak = max(range(i, j + 1), key=lambda k: abs(residuals[k]))  # type: ignore[arg-type]
        kind = AnomalyKind.SHIFT if j - i + 1 >= min_run else AnomalyKind.SPIKE
        anomalies.append(
            Anomaly(
                kind=kind,
                start_index=offset + i,
                end_index=offset + j,
                peak_index=offset + peak,
                max_residual=residuals[peak],  # type: ignore[arg-type]
            )
        )
        i = j + 1
    return anomalies

"""领域模型：以不可变数据类表达聚合，持久化层负责装配。

设计要点：
- 指标定义、换算规则均按版本登记，报告只固化版本号，历史报告不被改写；
- 观测数据只追加（迟到数据形成新数据版本），快照按版本可重放；
- 计算任务记录断点，报告记录输入指纹以支持复算核对。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .anomaly import AnomalyKind, AnomalyVerdict


class MissingPolicy(str, Enum):
    """缺失值处理策略。"""

    SKIP = "skip"  # 跳过缺失期间
    ZERO = "zero"  # 缺失按 0 计
    FAIL = "fail"  # 任一缺失即失败


class RuleStatus(str, Enum):
    """换算规则版本的生命周期。"""

    PENDING = "pending"  # 会签中
    ACTIVE = "active"  # 当前生效
    SUPERSEDED = "superseded"  # 被更新版本取代
    ROLLED_BACK = "rolled_back"  # 因回滚而失效


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ReportStatus(str, Enum):
    COMPUTED = "computed"  # 已计算，待复核
    REVIEWED = "reviewed"  # 复核通过，可导出
    REJECTED = "rejected"  # 复核驳回（终态）


# 计算任务的断点步骤，顺序即执行顺序。
CALCULATION_STEPS: tuple[str, ...] = ("snapshot", "convert", "aggregate", "persist")


@dataclass(frozen=True)
class Indicator:
    code: str
    name: str
    category: str  # 招生 / 就业 / 师资培养 等
    unit: str
    created_at: str


@dataclass(frozen=True)
class IndicatorVersion:
    """指标定义的一个版本；formula 为 JSON 可序列化的计算规格。"""

    code: str
    version_no: int
    formula: dict
    missing_policy: MissingPolicy
    created_at: str


@dataclass(frozen=True)
class EvidenceSource:
    """证据来源：观测记录必须挂接已登记证据。"""

    id: str
    project_id: str
    kind: str
    uri: str
    sha256: str
    registered_by: str
    registered_at: str


@dataclass(frozen=True)
class ImportBatch:
    """一次数据导入即项目的一个新数据版本。"""

    id: str
    project_id: str
    seq: int  # 数据版本号，按项目递增
    reason: str
    created_by: str
    created_at: str


@dataclass(frozen=True)
class Observation:
    """原始观测记录（追加日志中的一行）。"""

    batch_id: str
    project_id: str
    measure: str
    period: str
    caliber: str  # 数据来源国口径
    value: float | None  # None 表示缺失值
    retracted: bool  # True 表示该自然键被撤回
    evidence_id: str
    institution_id: str
    created_at: str

    @property
    def natural_key(self) -> tuple[str, str, str, str]:
        return (self.project_id, self.measure, self.period, self.caliber)


@dataclass(frozen=True)
class SnapshotRow:
    """某数据版本下的有效观测。"""

    measure: str
    period: str
    caliber: str
    value: float | None
    evidence_id: str


@dataclass(frozen=True)
class ConversionRule:
    """换算规则的一个版本：canonical = value * factor + offset。"""

    id: str
    rule_key: str  # measure|from_caliber|to_caliber
    version_no: int
    measure: str
    from_caliber: str
    to_caliber: str
    factor: float
    offset: float
    status: RuleStatus
    required_signatories: tuple[str, ...]
    created_by: str
    created_at: str


@dataclass(frozen=True)
class ComputationTask:
    id: str
    idempotency_key: str
    project_id: str
    window_start: str
    window_end: str
    target_caliber: str
    request_fingerprint: str
    status: TaskStatus
    current_step: str | None
    report_id: str | None
    error: str | None
    created_by: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Report:
    """项目结论：pins 固化全部输入版本，fingerprint 支持复算核对。"""

    id: str
    project_id: str
    window_start: str
    window_end: str
    target_caliber: str
    data_version_no: int
    pins: dict  # {"indicators": {code: ver}, "rules": {rule_key: ver}, "data_version": n}
    lines: list[dict]
    input_fingerprint: str
    result_fingerprint: str
    status: ReportStatus
    created_by: str
    created_at: str
    task_id: str


@dataclass(frozen=True)
class Grant:
    """授权粒度：机构 × 项目 × 指标类别 × 权限。"""

    institution_id: str
    project_id: str  # "*" 表示全部项目
    category: str  # "*" 表示全部类别
    permission: str  # import / view / calculate / review / export


@dataclass(frozen=True)
class Principal:
    """接口层解析出的调用者。"""

    institution_id: str
    role: str = "officer"  # supervisor 为主管单位，跳过授权检查

    @property
    def is_supervisor(self) -> bool:
        return self.role == "supervisor"


@dataclass(frozen=True)
class AnomalyRule:
    """趋势异常监测规则的一个版本；同 rule_key 递增版本号，最新版生效。"""

    id: str
    rule_key: str  # 监测对象键，如 "proj-1|enrollment_count|CN-STD"
    version_no: int
    baseline_window: int
    sigma_threshold: float
    min_run: int
    created_by: str
    created_at: str


@dataclass(frozen=True)
class AnomalyEvaluation:
    """一次趋势判定：固化规则快照与原始序列，规则变更后判定依据仍可查。"""

    id: str
    rule_id: str
    rule_key: str
    rule_version_no: int
    rule_snapshot: dict  # 判定时的规则参数全文（判定依据）
    series: list  # 原始序列 [float | None]
    series_fingerprint: str
    verdict: AnomalyVerdict
    created_by: str
    created_at: str


@dataclass(frozen=True)
class AnomalyAlert:
    """判定产生的一条告警（尖峰或偏移）；无异常的判定不产生告警行。"""

    id: str
    evaluation_id: str
    kind: AnomalyKind
    start_index: int
    end_index: int
    peak_index: int
    max_residual: float
    created_at: str

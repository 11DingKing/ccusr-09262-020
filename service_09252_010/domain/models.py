"""领域模型：以不可变数据类表达聚合，持久化层负责装配。

设计要点：
- 指标定义、换算规则均按版本登记，报告只固化版本号，历史报告不被改写；
- 观测数据只追加（迟到数据形成新数据版本），快照按版本可重放；
- 计算任务记录断点，报告记录输入指纹以支持复算核对。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .trend import AnomalyKind, TrendRuleSpec


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
class TrendRule:
    """趋势监测规则的一个版本；只增不改，告警固化版本号与参数快照。"""

    id: str
    rule_key: str  # 监测对象（指标/度量名）
    version_no: int
    window: int  # 基线参考期长度
    z_threshold: float  # 偏离分数阈值
    min_run: int  # 持续偏移所需的最短连续偏离长度
    created_by: str
    created_at: str

    def spec(self) -> TrendRuleSpec:
        """判定参数视图，供纯函数检测使用。"""
        return TrendRuleSpec(
            window=self.window,
            z_threshold=self.z_threshold,
            min_run=self.min_run,
        )


@dataclass(frozen=True)
class TrendSeries:
    """上报的原始监测序列（None 为缺失点）。"""

    id: str
    metric: str
    points: list[float | None]
    created_by: str
    created_at: str


@dataclass(frozen=True)
class TrendAlert:
    """一条趋势异常告警：判定时的规则版本与参数快照一并固化。

    规则变更后，历史告警仍保留原判定依据，不被改写。
    """

    id: str
    series_id: str
    metric: str
    rule_id: str
    rule_version_no: int
    rule_snapshot: dict  # {"window":…, "z_threshold":…, "min_run":…}
    kind: AnomalyKind  # spike 一次尖峰 / shift 持续偏移
    start_index: int  # 事件起始点在原始序列中的下标
    length: int  # 连续偏离点数
    peak_value: float
    peak_score: float
    baseline: float
    created_by: str
    created_at: str

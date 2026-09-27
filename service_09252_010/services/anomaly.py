"""指标趋势异常监测服务：规则版本化登记、序列判定与告警留痕。

- 规则按版本登记（同 rule_key 递增版本号），判定始终使用最新版本；
- 每次判定把**规则快照**（判定依据）与**原始序列**一并固化到 SQLite，
  规则变更后历史告警仍保留原判定依据；
- 判定算法见 domain.anomaly：区分一次尖峰（spike）与持续偏移（shift），
  空序列不产生告警。
"""
from __future__ import annotations

from ..domain.anomaly import AnomalyRuleSpec, detect_anomalies
from ..domain.errors import NotFoundError, PermissionDeniedError, ValidationError
from ..domain.fingerprint import fingerprint
from ..domain.models import (
    AnomalyAlert,
    AnomalyEvaluation,
    AnomalyRule,
    Principal,
)
from ..persistence.database import Database
from ..persistence.store import Store
from ..ports import Clock, IdGenerator


class AnomalyService:
    def __init__(self, db: Database, clock: Clock, ids: IdGenerator) -> None:
        self.db = db
        self.clock = clock
        self.ids = ids

    def register_rule(self, principal: Principal, *, rule_key: str,
                      baseline_window: int = 5, sigma_threshold: float = 3.0,
                      min_run: int = 3) -> dict:
        """登记监测规则新版本。仅主管单位可登记。"""
        if not principal.is_supervisor:
            raise PermissionDeniedError("仅主管单位可登记趋势监测规则")
        if not isinstance(rule_key, str) or not rule_key.strip():
            raise ValidationError("rule_key 必须为非空字符串")
        if isinstance(baseline_window, bool) or not isinstance(baseline_window, int) \
                or baseline_window < 2:
            raise ValidationError("baseline_window 必须为不小于 2 的整数")
        if isinstance(sigma_threshold, bool) \
                or not isinstance(sigma_threshold, (int, float)) \
                or sigma_threshold <= 0:
            raise ValidationError("sigma_threshold 必须为正数")
        if isinstance(min_run, bool) or not isinstance(min_run, int) or min_run < 1:
            raise ValidationError("min_run 必须为不小于 1 的整数")
        with self.db.uow() as uow:
            store = Store(uow.conn)
            version_no = store.next_anomaly_rule_version(rule_key)
            rule = AnomalyRule(
                id=self.ids.new_id("arule"),
                rule_key=rule_key,
                version_no=version_no,
                baseline_window=baseline_window,
                sigma_threshold=float(sigma_threshold),
                min_run=min_run,
                created_by=principal.institution_id,
                created_at=self.clock.now(),
            )
            store.add_anomaly_rule(rule)
        return {"rule_id": rule.id, "rule_key": rule_key,
                "version_no": version_no}

    def evaluate(self, principal: Principal, *, rule_key: str,
                 values: list) -> dict:
        """按当前规则版本判定序列，固化判定依据与原始序列并生成告警。"""
        series = self._validate_series(values)
        with self.db.uow() as uow:
            store = Store(uow.conn)
            rule = store.latest_anomaly_rule(rule_key)
            if rule is None:
                raise NotFoundError(f"趋势监测规则不存在: {rule_key}")
            spec = AnomalyRuleSpec(
                baseline_window=rule.baseline_window,
                sigma_threshold=rule.sigma_threshold,
                min_run=rule.min_run,
            )
            judgment = detect_anomalies(series, spec)
            now = self.clock.now()
            evaluation = AnomalyEvaluation(
                id=self.ids.new_id("aev"),
                rule_id=rule.id,
                rule_key=rule.rule_key,
                rule_version_no=rule.version_no,
                rule_snapshot={
                    "rule_key": rule.rule_key,
                    "version_no": rule.version_no,
                    "baseline_window": rule.baseline_window,
                    "sigma_threshold": rule.sigma_threshold,
                    "min_run": rule.min_run,
                },
                series=series,
                series_fingerprint=fingerprint(series),
                verdict=judgment.verdict,
                created_by=principal.institution_id,
                created_at=now,
            )
            store.add_anomaly_evaluation(evaluation)
            alerts = [
                AnomalyAlert(
                    id=self.ids.new_id("aal"),
                    evaluation_id=evaluation.id,
                    kind=anomaly.kind,
                    start_index=anomaly.start_index,
                    end_index=anomaly.end_index,
                    peak_index=anomaly.peak_index,
                    max_residual=anomaly.max_residual,
                    created_at=now,
                )
                for anomaly in judgment.anomalies
            ]
            for alert in alerts:
                store.add_anomaly_alert(alert)
        return {
            "evaluation_id": evaluation.id,
            "rule_key": rule.rule_key,
            "rule_version_no": rule.version_no,
            "verdict": judgment.verdict.value,
            "alerts": [self._alert_payload(a) for a in alerts],
        }

    def get_evaluation(self, evaluation_id: str) -> dict:
        """判定详情：含固化的规则快照（原判定依据）、原始序列与告警。"""
        with self.db.read() as conn:
            store = Store(conn)
            ev = store.get_anomaly_evaluation(evaluation_id)
            if ev is None:
                raise NotFoundError(f"趋势判定不存在: {evaluation_id}")
            alerts = store.list_anomaly_alerts(ev.id)
        return {
            "evaluation_id": ev.id,
            "rule_key": ev.rule_key,
            "rule_version_no": ev.rule_version_no,
            "rule_snapshot": ev.rule_snapshot,
            "series": ev.series,
            "series_fingerprint": ev.series_fingerprint,
            "verdict": ev.verdict.value,
            "created_by": ev.created_by,
            "created_at": ev.created_at,
            "alerts": [self._alert_payload(a) for a in alerts],
        }

    def list_evaluations(self, rule_key: str | None = None) -> list[dict]:
        with self.db.read() as conn:
            store = Store(conn)
            evaluations = store.list_anomaly_evaluations(rule_key)
            counts = {
                ev.id: len(store.list_anomaly_alerts(ev.id))
                for ev in evaluations
            }
        return [
            {
                "evaluation_id": ev.id,
                "rule_key": ev.rule_key,
                "rule_version_no": ev.rule_version_no,
                "verdict": ev.verdict.value,
                "alert_count": counts[ev.id],
                "created_by": ev.created_by,
                "created_at": ev.created_at,
            }
            for ev in evaluations
        ]

    def list_rules(self) -> list[dict]:
        with self.db.read() as conn:
            rules = Store(conn).list_anomaly_rules()
        latest: dict[str, int] = {}
        for rule in rules:
            latest[rule.rule_key] = max(latest.get(rule.rule_key, 0),
                                        rule.version_no)
        return [
            {
                "rule_id": r.id,
                "rule_key": r.rule_key,
                "version_no": r.version_no,
                "baseline_window": r.baseline_window,
                "sigma_threshold": r.sigma_threshold,
                "min_run": r.min_run,
                "is_current": latest[r.rule_key] == r.version_no,
            }
            for r in rules
        ]

    @staticmethod
    def _validate_series(values: list) -> list:
        if not isinstance(values, (list, tuple)):
            raise ValidationError("values 必须为数值序列（缺失用 null 表示）")
        series = []
        for value in values:
            if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))):
                raise ValidationError(
                    f"序列元素必须为数值或 null: {value!r}"
                )
            series.append(None if value is None else float(value))
        return series

    @staticmethod
    def _alert_payload(alert: AnomalyAlert) -> dict:
        return {
            "alert_id": alert.id,
            "kind": alert.kind.value,
            "start_index": alert.start_index,
            "end_index": alert.end_index,
            "peak_index": alert.peak_index,
            "max_residual": alert.max_residual,
            "direction": "up" if alert.max_residual > 0 else "down",
        }

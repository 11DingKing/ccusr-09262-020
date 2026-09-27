"""指标趋势异常监测服务。

- 监测规则按版本登记、只增不改；评估始终使用当前最新版本；
- 告警落库时固化规则版本号与参数快照，规则变更后历史告警的
  判定依据不变、不被改写；
- 原始序列与告警判定都存 SQLite，可复核、可复算；
- 判定区分一次尖峰（spike）与持续偏移（shift），见 domain.trend；
- 空序列与纯重复数据不会产生告警（杜绝误报）；
- 监测不改动项目数据，仅追加监测记录，故上报/评估/查询不限角色，
  规则维护仅主管单位。
"""
from __future__ import annotations

import math

from ..domain.errors import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from ..domain.models import Principal, TrendAlert, TrendRule, TrendSeries
from ..domain.trend import TrendRuleSpec, detect_anomalies, validate_spec
from ..persistence.database import Database
from ..persistence.store import Store
from ..ports import Clock, IdGenerator


class TrendMonitorService:
    def __init__(self, db: Database, clock: Clock, ids: IdGenerator) -> None:
        self.db = db
        self.clock = clock
        self.ids = ids

    # ---- 规则版本 ----
    def register_rule(self, principal: Principal, *, rule_key: str,
                      window: int, z_threshold: float, min_run: int) -> dict:
        """登记监测规则首个版本（v1）。仅主管单位可登记。"""
        self._require_supervisor(principal)
        key = self._check_name(rule_key, "rule_key")
        self._check_params(window, z_threshold, min_run)
        with self.db.uow() as uow:
            store = Store(uow.conn)
            if store.latest_trend_rule(key) is not None:
                raise ConflictError(f"监测规则已存在: {key}，请登记新版本")
            rule = TrendRule(
                self.ids.new_id("trule"), key, 1, window, float(z_threshold),
                min_run, principal.institution_id, self.clock.now(),
            )
            store.add_trend_rule(rule)
        return {"rule_id": rule.id, "rule_key": key, "version_no": 1}

    def add_rule_version(self, principal: Principal, rule_key: str, *,
                         window: int, z_threshold: float,
                         min_run: int) -> dict:
        """登记规则新版本；历史告警保留原判定依据，不受影响。"""
        self._require_supervisor(principal)
        key = self._check_name(rule_key, "rule_key")
        self._check_params(window, z_threshold, min_run)
        with self.db.uow() as uow:
            store = Store(uow.conn)
            latest = store.latest_trend_rule(key)
            if latest is None:
                raise NotFoundError(f"监测规则不存在: {key}")
            rule = TrendRule(
                self.ids.new_id("trule"), key, latest.version_no + 1, window,
                float(z_threshold), min_run, principal.institution_id,
                self.clock.now(),
            )
            store.add_trend_rule(rule)
        return {"rule_id": rule.id, "rule_key": key,
                "version_no": rule.version_no}

    def get_rule(self, rule_key: str) -> dict:
        """规则详情（含全部版本）。"""
        with self.db.read() as conn:
            versions = Store(conn).list_trend_rule_versions(rule_key)
        if not versions:
            raise NotFoundError(f"监测规则不存在: {rule_key}")
        return {
            "rule_key": rule_key,
            "versions": [
                {
                    "rule_id": v.id,
                    "version_no": v.version_no,
                    "window": v.window,
                    "z_threshold": v.z_threshold,
                    "min_run": v.min_run,
                    "created_by": v.created_by,
                    "created_at": v.created_at,
                }
                for v in versions
            ],
        }

    # ---- 原始序列 ----
    def submit_series(self, principal: Principal, *, metric: str,
                      points: list) -> dict:
        """登记原始监测序列；空序列允许（评估时不会产生告警）。"""
        name = self._check_name(metric, "metric")
        pts = self._check_points(points)
        series = TrendSeries(
            self.ids.new_id("tseries"), name, pts,
            principal.institution_id, self.clock.now(),
        )
        with self.db.uow() as uow:
            Store(uow.conn).add_trend_series(series)
        return {"series_id": series.id, "metric": name, "length": len(pts)}

    def get_series(self, series_id: str) -> dict:
        """原始序列详情。"""
        with self.db.read() as conn:
            series = Store(conn).get_trend_series(series_id)
        if series is None:
            raise NotFoundError(f"序列不存在: {series_id}")
        return {
            "series_id": series.id,
            "metric": series.metric,
            "points": series.points,
            "created_by": series.created_by,
            "created_at": series.created_at,
        }

    # ---- 评估与告警 ----
    def evaluate(self, principal: Principal, series_id: str) -> dict:
        """按当前最新规则版本判定序列并落库告警。

        同一序列在同一规则版本下重复评估收敛为同一批告警（幂等）；
        规则升版后再次评估则按新版本另出判定，历史告警保留。
        """
        with self.db.uow() as uow:
            store = Store(uow.conn)
            series = store.get_trend_series(series_id)
            if series is None:
                raise NotFoundError(f"序列不存在: {series_id}")
            rule = store.latest_trend_rule(series.metric)
            if rule is None:
                raise NotFoundError(f"指标未登记监测规则: {series.metric}")
            alerts = store.trend_alerts_for(series.id, rule.id)
            if not alerts:
                snapshot = {
                    "window": rule.window,
                    "z_threshold": rule.z_threshold,
                    "min_run": rule.min_run,
                }
                now = self.clock.now()
                for anomaly in detect_anomalies(series.points, rule.spec()):
                    alert = TrendAlert(
                        self.ids.new_id("talert"), series.id, series.metric,
                        rule.id, rule.version_no, snapshot, anomaly.kind,
                        anomaly.index, anomaly.length, anomaly.peak_value,
                        anomaly.peak_score, anomaly.baseline,
                        principal.institution_id, now,
                    )
                    store.add_trend_alert(alert)
                    alerts.append(alert)
        return {
            "series_id": series.id,
            "metric": series.metric,
            "rule_version_no": rule.version_no,
            "alerts": [self._alert_dict(a) for a in alerts],
        }

    def list_alerts(self, *, metric: str | None = None) -> list[dict]:
        """告警列表；每条都带判定时的规则快照（历史判定依据）。"""
        with self.db.read() as conn:
            alerts = Store(conn).list_trend_alerts(metric)
        return [self._alert_dict(a) for a in alerts]

    # ---- 内部 ----
    @staticmethod
    def _alert_dict(alert: TrendAlert) -> dict:
        return {
            "alert_id": alert.id,
            "series_id": alert.series_id,
            "metric": alert.metric,
            "kind": alert.kind.value,
            "start_index": alert.start_index,
            "length": alert.length,
            "peak_value": alert.peak_value,
            "peak_score": alert.peak_score,
            "baseline": alert.baseline,
            "rule_id": alert.rule_id,
            "rule_version_no": alert.rule_version_no,
            "rule_snapshot": alert.rule_snapshot,
            "created_by": alert.created_by,
            "created_at": alert.created_at,
        }

    @staticmethod
    def _require_supervisor(principal: Principal) -> None:
        if not principal.is_supervisor:
            raise PermissionDeniedError("仅主管单位可维护监测规则")

    @staticmethod
    def _check_params(window: int, z_threshold: float,
                      min_run: int) -> None:
        try:
            validate_spec(TrendRuleSpec(window, z_threshold, min_run))
        except ValueError as exc:
            raise ValidationError(str(exc)) from None

    @staticmethod
    def _check_name(raw: object, label: str) -> str:
        if not isinstance(raw, str) or not raw.strip():
            raise ValidationError(f"{label} 必须为非空字符串")
        return raw.strip()

    @staticmethod
    def _check_points(points: object) -> list[float | None]:
        if not isinstance(points, (list, tuple)):
            raise ValidationError("points 必须为数组，元素为数值或 null")
        checked: list[float | None] = []
        for p in points:
            if p is None:
                checked.append(None)
                continue
            if isinstance(p, bool) or not isinstance(p, (int, float)):
                raise ValidationError(f"序列点必须为数值或 null: {p!r}")
            value = float(p)
            if not math.isfinite(value):
                raise ValidationError(f"序列点必须为有限数值: {p!r}")
            checked.append(value)
        return checked

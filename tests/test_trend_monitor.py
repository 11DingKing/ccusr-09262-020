"""指标趋势异常监测：尖峰/偏移区分、空序列与重复数据无误报、规则变更留痕。"""
from __future__ import annotations

import math
import unittest

from service_09252_010.domain.errors import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from service_09252_010.domain.trend import (
    AnomalyKind,
    TrendRuleSpec,
    detect_anomalies,
)
from support import INST_A, SUPERVISOR, RigTestCase

SPEC = TrendRuleSpec(window=4, z_threshold=2.0, min_run=3)
# 基线 [10, 12, 10, 12]：均值 11，样本标准差 ≈1.1547；
# 取值 14 的偏离分数 ≈2.598（过阈值 2.0、不过 3.0），取值 11 不偏离。
BASELINE = [10.0, 12.0, 10.0, 12.0]


class DetectTests(unittest.TestCase):
    """纯函数判定：不触库。"""

    def test_empty_series_yields_nothing(self) -> None:
        self.assertEqual(detect_anomalies([], SPEC), [])

    def test_series_shorter_than_window_yields_nothing(self) -> None:
        self.assertEqual(detect_anomalies([1.0, 2.0], SPEC), [])
        self.assertEqual(detect_anomalies([None, None], SPEC), [])

    def test_duplicate_data_yields_nothing(self) -> None:
        self.assertEqual(detect_anomalies([7.0] * 12, SPEC), [])

    def test_single_spike(self) -> None:
        anomalies = detect_anomalies(BASELINE + [14.0, 11.0, 10.0], SPEC)
        self.assertEqual(len(anomalies), 1)
        spike = anomalies[0]
        self.assertEqual(spike.kind, AnomalyKind.SPIKE)
        self.assertEqual((spike.index, spike.length), (4, 1))
        self.assertEqual(spike.peak_value, 14.0)
        self.assertAlmostEqual(spike.peak_score, 3 / math.sqrt(4 / 3), places=6)
        self.assertEqual(spike.baseline, 11.0)

    def test_sustained_shift(self) -> None:
        anomalies = detect_anomalies(BASELINE + [14.0, 14.0, 14.0, 11.0], SPEC)
        self.assertEqual(len(anomalies), 1)
        shift = anomalies[0]
        self.assertEqual(shift.kind, AnomalyKind.SHIFT)
        self.assertEqual((shift.index, shift.length), (4, 3))

    def test_run_below_min_run_is_spike(self) -> None:
        anomalies = detect_anomalies(BASELINE + [14.0, 14.0, 11.0], SPEC)
        self.assertEqual([a.kind for a in anomalies], [AnomalyKind.SPIKE])
        self.assertEqual(anomalies[0].length, 2)

    def test_missing_point_breaks_run(self) -> None:
        anomalies = detect_anomalies(BASELINE + [14.0, None, 14.0, 14.0], SPEC)
        self.assertEqual([a.kind for a in anomalies],
                         [AnomalyKind.SPIKE, AnomalyKind.SPIKE])

    def test_zero_variance_baseline_flags_any_change(self) -> None:
        anomalies = detect_anomalies([5.0] * 5 + [6.0], SPEC)
        self.assertEqual(len(anomalies), 1)
        self.assertTrue(math.isinf(anomalies[0].peak_score))

    def test_invalid_spec_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detect_anomalies([1.0] * 9, TrendRuleSpec(1, 2.0, 3))
        with self.assertRaises(ValueError):
            detect_anomalies([1.0] * 9, TrendRuleSpec(4, 0.0, 3))
        with self.assertRaises(ValueError):
            detect_anomalies([1.0] * 9, TrendRuleSpec(4, 2.0, 1))


class TrendMonitorServiceTests(RigTestCase):
    def _register(self, z_threshold: float = 2.0, min_run: int = 3) -> None:
        self.rig.trend.register_rule(
            SUPERVISOR, rule_key="enrollment_total",
            window=4, z_threshold=z_threshold, min_run=min_run,
        )

    def _submit(self, points: list, metric: str = "enrollment_total") -> str:
        return self.rig.trend.submit_series(
            INST_A, metric=metric, points=points
        )["series_id"]

    def test_empty_series_no_false_alarm(self) -> None:
        self._register()
        series_id = self._submit([])
        result = self.rig.trend.evaluate(INST_A, series_id)
        self.assertEqual(result["alerts"], [])
        self.assertEqual(self.rig.trend.list_alerts(), [])
        # 原始序列仍完整留存
        self.assertEqual(self.rig.trend.get_series(series_id)["points"], [])

    def test_duplicate_data_no_false_alarm(self) -> None:
        self._register()
        series_id = self._submit([5.0] * 12)
        self.assertEqual(self.rig.trend.evaluate(INST_A, series_id)["alerts"], [])
        self.assertEqual(
            self.rig.trend.list_alerts(metric="enrollment_total"), []
        )

    def test_spike_and_shift_classified(self) -> None:
        self._register()
        spike_id = self._submit(BASELINE + [14.0, 11.0])
        shift_id = self._submit(BASELINE + [14.0, 14.0, 14.0])
        spike = self.rig.trend.evaluate(INST_A, spike_id)["alerts"]
        shift = self.rig.trend.evaluate(INST_A, shift_id)["alerts"]
        self.assertEqual([a["kind"] for a in spike], ["spike"])
        self.assertEqual([a["kind"] for a in shift], ["shift"])
        self.assertEqual(spike[0]["rule_version_no"], 1)
        # 判定依据随告警固化
        self.assertEqual(
            spike[0]["rule_snapshot"],
            {"window": 4, "z_threshold": 2.0, "min_run": 3},
        )

    def test_alerts_and_series_persisted(self) -> None:
        self._register()
        points = BASELINE + [14.0, 14.0, 14.0]
        series_id = self._submit(points)
        self.rig.trend.evaluate(INST_A, series_id)
        self.assertEqual(self.rig.trend.get_series(series_id)["points"], points)
        alerts = self.rig.trend.list_alerts(metric="enrollment_total")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["series_id"], series_id)
        self.assertEqual(alerts[0]["kind"], "shift")

    def test_rule_change_keeps_historical_basis(self) -> None:
        self._register(z_threshold=2.0)  # v1
        points = BASELINE + [14.0, 11.0]  # 14 的偏离分数 ≈2.598
        series_a = self._submit(points)
        first = self.rig.trend.evaluate(INST_A, series_a)["alerts"]
        self.assertEqual(len(first), 1)
        # 规则变更：阈值上调到 3.0（v2），同样的数据不再构成告警
        self.rig.trend.add_rule_version(
            SUPERVISOR, "enrollment_total",
            window=4, z_threshold=3.0, min_run=3,
        )
        # 历史告警仍保留 v1 的判定依据
        historical = self.rig.trend.list_alerts(metric="enrollment_total")
        self.assertEqual(len(historical), 1)
        self.assertEqual(historical[0]["rule_version_no"], 1)
        self.assertEqual(historical[0]["rule_snapshot"]["z_threshold"], 2.0)
        # 同一序列按 v2 复评：不再产生新告警，历史告警不被改写
        second = self.rig.trend.evaluate(INST_A, series_a)
        self.assertEqual(second["rule_version_no"], 2)
        self.assertEqual(second["alerts"], [])
        self.assertEqual(len(self.rig.trend.list_alerts()), 1)
        # 新序列同样按 v2 判定
        series_b = self._submit(points)
        self.assertEqual(self.rig.trend.evaluate(INST_A, series_b)["alerts"], [])

    def test_evaluate_is_idempotent(self) -> None:
        self._register()
        series_id = self._submit(BASELINE + [14.0, 14.0, 14.0])
        first = self.rig.trend.evaluate(INST_A, series_id)
        again = self.rig.trend.evaluate(INST_A, series_id)
        self.assertEqual(
            [a["alert_id"] for a in again["alerts"]],
            [a["alert_id"] for a in first["alerts"]],
        )
        self.assertEqual(len(self.rig.trend.list_alerts()), 1)

    def test_rule_versions_accumulate(self) -> None:
        self._register()
        self.rig.trend.add_rule_version(
            SUPERVISOR, "enrollment_total",
            window=6, z_threshold=2.5, min_run=4,
        )
        rule = self.rig.trend.get_rule("enrollment_total")
        self.assertEqual([v["version_no"] for v in rule["versions"]], [1, 2])
        self.assertEqual(rule["versions"][1]["window"], 6)

    def test_rule_registration_validation(self) -> None:
        with self.assertRaises(PermissionDeniedError):
            self.rig.trend.register_rule(
                INST_A, rule_key="m", window=4, z_threshold=2.0, min_run=3
            )
        with self.assertRaises(ValidationError):
            self.rig.trend.register_rule(
                SUPERVISOR, rule_key="m", window=1, z_threshold=2.0, min_run=3
            )
        with self.assertRaises(ValidationError):
            self.rig.trend.register_rule(
                SUPERVISOR, rule_key="m", window=4, z_threshold=0.0, min_run=3
            )
        with self.assertRaises(ValidationError):
            self.rig.trend.register_rule(
                SUPERVISOR, rule_key="m", window=4, z_threshold=2.0, min_run=1
            )
        self._register()
        with self.assertRaises(ConflictError):
            self._register()
        with self.assertRaises(NotFoundError):
            self.rig.trend.add_rule_version(
                SUPERVISOR, "ghost", window=4, z_threshold=2.0, min_run=3
            )

    def test_series_validation(self) -> None:
        self._register()
        with self.assertRaises(ValidationError):
            self._submit([1.0, float("nan"), 2.0])
        with self.assertRaises(ValidationError):
            self._submit([1.0, "2"])
        with self.assertRaises(ValidationError):
            self._submit([1.0, True])

    def test_unknown_series_and_missing_rule(self) -> None:
        self._register()
        with self.assertRaises(NotFoundError):
            self.rig.trend.evaluate(INST_A, "tseries-9999")
        other = self._submit([1.0] * 6, metric="unmonitored_metric")
        with self.assertRaises(NotFoundError):
            self.rig.trend.evaluate(INST_A, other)


if __name__ == "__main__":
    unittest.main()

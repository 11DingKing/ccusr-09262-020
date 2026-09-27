"""指标趋势异常监测：尖峰与偏移的区分、判定留痕、空序列与重复数据。"""
from __future__ import annotations

import unittest

from service_09252_010.domain.anomaly import (
    AnomalyKind,
    AnomalyRuleSpec,
    AnomalyVerdict,
    detect_anomalies,
)
from service_09252_010.domain.errors import (
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from service_09252_010.domain.fingerprint import fingerprint
from service_09252_010.persistence.store import Store
from support import INST_A, RigTestCase, SUPERVISOR

RULE_KEY = "proj-ivc-01|enrollment_count|CN-STD"
SPEC = AnomalyRuleSpec(baseline_window=4, sigma_threshold=3.0, min_run=3)

# 基线 [10, 11, 9, 10]：中位数 10，MAD 0.5，scale ≈ 0.7413；
# 值 30 的稳健残差 ≈ 27σ，值 12 ≈ 2.7σ（不越限）。
SPIKE_SERIES = [10, 11, 9, 10, 30, 10, 11]
SHIFT_SERIES = [10, 11, 9, 10, 30, 29, 31, 10]


class DetectAnomaliesTests(unittest.TestCase):
    """纯领域判定：不依赖数据库。"""

    def test_empty_sequence_yields_no_anomaly(self) -> None:
        """空序列不会生成误报。"""
        judgment = detect_anomalies([], SPEC)
        self.assertEqual(judgment.verdict, AnomalyVerdict.NORMAL)
        self.assertEqual(judgment.anomalies, ())

    def test_single_point_and_short_series_yield_no_anomaly(self) -> None:
        for series in ([5], [10, 11, 9], [10, 11, 9, 10]):
            judgment = detect_anomalies(series, SPEC)
            self.assertEqual(judgment.verdict, AnomalyVerdict.NORMAL, series)

    def test_spike_distinguished_from_shift(self) -> None:
        spike = detect_anomalies(SPIKE_SERIES, SPEC)
        self.assertEqual(spike.verdict, AnomalyVerdict.SPIKE)
        self.assertEqual(len(spike.anomalies), 1)
        anomaly = spike.anomalies[0]
        self.assertEqual(anomaly.kind, AnomalyKind.SPIKE)
        self.assertEqual((anomaly.start_index, anomaly.end_index), (4, 4))
        self.assertEqual(anomaly.direction, "up")

        shift = detect_anomalies(SHIFT_SERIES, SPEC)
        self.assertEqual(shift.verdict, AnomalyVerdict.SHIFT)
        self.assertEqual(shift.anomalies[0].kind, AnomalyKind.SHIFT)
        self.assertEqual(
            (shift.anomalies[0].start_index, shift.anomalies[0].end_index),
            (4, 6),
        )

    def test_run_length_boundary_at_min_run(self) -> None:
        """连续 2 点（不足 min_run=3）判尖峰，连续 3 点判偏移。"""
        two = detect_anomalies([10, 11, 9, 10, 30, 29, 10], SPEC)
        self.assertEqual(two.anomalies[0].kind, AnomalyKind.SPIKE)
        three = detect_anomalies(SHIFT_SERIES, SPEC)
        self.assertEqual(three.anomalies[0].kind, AnomalyKind.SHIFT)

    def test_downward_shift_direction(self) -> None:
        judgment = detect_anomalies([10, 11, 9, 10, 2, 3, 1, 10], SPEC)
        anomaly = judgment.anomalies[0]
        self.assertEqual(anomaly.kind, AnomalyKind.SHIFT)
        self.assertEqual(anomaly.direction, "down")
        self.assertLess(anomaly.max_residual, 0)

    def test_min_run_one_makes_every_run_a_shift(self) -> None:
        spec = AnomalyRuleSpec(baseline_window=4, sigma_threshold=3.0, min_run=1)
        judgment = detect_anomalies(SPIKE_SERIES, spec)
        self.assertEqual(judgment.anomalies[0].kind, AnomalyKind.SHIFT)

    def test_none_breaks_run_and_is_never_anomalous(self) -> None:
        series = [10, 10, 10, 10, 30, None, 30, 30]
        judgment = detect_anomalies(series, SPEC)
        kinds = [a.kind for a in judgment.anomalies]
        self.assertEqual(kinds, [AnomalyKind.SPIKE, AnomalyKind.SPIKE])
        self.assertEqual(judgment.verdict, AnomalyVerdict.SPIKE)

    def test_all_none_series_yields_no_anomaly(self) -> None:
        judgment = detect_anomalies([None] * 8, SPEC)
        self.assertEqual(judgment.verdict, AnomalyVerdict.NORMAL)


class AnomalyServiceTests(RigTestCase):
    def register_rule(self, **overrides) -> dict:
        params = {"rule_key": RULE_KEY, "baseline_window": 4,
                  "sigma_threshold": 3.0, "min_run": 3}
        params.update(overrides)
        return self.rig.anomaly.register_rule(SUPERVISOR, **params)

    def test_spike_and_shift_are_distinguished_and_persisted(self) -> None:
        self.register_rule()
        spike = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY, values=SPIKE_SERIES)
        self.assertEqual(spike["verdict"], "spike")
        self.assertEqual(len(spike["alerts"]), 1)
        self.assertEqual(spike["alerts"][0]["kind"], "spike")
        self.assertEqual(spike["alerts"][0]["direction"], "up")

        shift = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY, values=SHIFT_SERIES)
        self.assertEqual(shift["verdict"], "shift")
        self.assertEqual(shift["alerts"][0]["kind"], "shift")
        self.assertEqual(
            (shift["alerts"][0]["start_index"],
             shift["alerts"][0]["end_index"]), (4, 6))

    def test_empty_sequence_generates_no_false_alarm(self) -> None:
        """空序列判定为正常、不产生告警，但判定与空序列仍留痕。"""
        self.register_rule()
        result = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY, values=[])
        self.assertEqual(result["verdict"], "normal")
        self.assertEqual(result["alerts"], [])

        detail = self.rig.anomaly.get_evaluation(result["evaluation_id"])
        self.assertEqual(detail["series"], [])
        self.assertEqual(detail["verdict"], "normal")
        with self.rig.db.read() as conn:
            store = Store(conn)
            self.assertEqual(
                store.list_anomaly_alerts(result["evaluation_id"]), [])
            self.assertIsNotNone(
                store.get_anomaly_evaluation(result["evaluation_id"]))

    def test_duplicate_data_generates_no_false_alarm(self) -> None:
        """重复数据（平坦序列）不触发误报。"""
        self.register_rule()
        result = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY, values=[7, 7, 7, 7, 7, 7, 7, 7])
        self.assertEqual(result["verdict"], "normal")
        self.assertEqual(result["alerts"], [])

    def test_duplicate_data_with_single_deviation_is_spike(self) -> None:
        """平坦基线下的单次偏离仍判为尖峰（尺度兜底不吞掉异常）。"""
        self.register_rule()
        result = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY, values=[5, 5, 5, 5, 5, 5, 9])
        self.assertEqual(result["verdict"], "spike")
        self.assertEqual(result["alerts"][0]["peak_index"], 6)

    def test_evaluation_persists_judgment_and_raw_series(self) -> None:
        """告警判定与原始序列都落 SQLite，可原样读回。"""
        self.register_rule()
        result = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY, values=SPIKE_SERIES)
        evaluation_id = result["evaluation_id"]
        with self.rig.db.read() as conn:
            store = Store(conn)
            ev = store.get_anomaly_evaluation(evaluation_id)
            assert ev is not None
            self.assertEqual(ev.series, [float(v) for v in SPIKE_SERIES])
            self.assertEqual(ev.series_fingerprint, fingerprint(ev.series))
            self.assertEqual(ev.verdict, AnomalyVerdict.SPIKE)
            alerts = store.list_anomaly_alerts(evaluation_id)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0].kind, AnomalyKind.SPIKE)
            self.assertEqual(alerts[0].evaluation_id, evaluation_id)

    def test_rule_change_keeps_historical_alert_basis(self) -> None:
        """规则变更后，历史告警仍保留原判定依据（规则快照与版本号）。"""
        first = self.register_rule()
        self.assertEqual(first["version_no"], 1)
        before = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY, values=SPIKE_SERIES)
        self.assertEqual(before["verdict"], "spike")

        # 阈值大幅收紧后的新版本：同一序列不再报警
        second = self.register_rule(sigma_threshold=1000.0)
        self.assertEqual(second["version_no"], 2)
        after = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY, values=SPIKE_SERIES)
        self.assertEqual(after["rule_version_no"], 2)
        self.assertEqual(after["verdict"], "normal")
        self.assertEqual(after["alerts"], [])

        # 历史判定仍按 v1 的依据可查，告警不被动摇
        detail = self.rig.anomaly.get_evaluation(before["evaluation_id"])
        self.assertEqual(detail["rule_version_no"], 1)
        self.assertEqual(detail["rule_snapshot"]["sigma_threshold"], 3.0)
        self.assertEqual(detail["rule_snapshot"]["min_run"], 3)
        self.assertEqual(detail["verdict"], "spike")
        self.assertEqual(len(detail["alerts"]), 1)
        self.assertEqual(detail["alerts"][0]["kind"], "spike")

        evaluations = self.rig.anomaly.list_evaluations(RULE_KEY)
        self.assertEqual(
            [(e["rule_version_no"], e["verdict"]) for e in evaluations],
            [(1, "spike"), (2, "normal")],
        )

    def test_list_rules_marks_current_version(self) -> None:
        self.register_rule()
        self.register_rule(sigma_threshold=4.5)
        rules = self.rig.anomaly.list_rules()
        self.assertEqual([(r["version_no"], r["is_current"]) for r in rules],
                         [(1, False), (2, True)])

    def test_missing_values_never_anomalous_and_break_runs(self) -> None:
        self.register_rule()
        result = self.rig.anomaly.evaluate(
            INST_A, rule_key=RULE_KEY,
            values=[10, 10, 10, 10, 30, None, 30, 30])
        self.assertEqual(result["verdict"], "spike")
        self.assertEqual(len(result["alerts"]), 2)
        self.assertTrue(all(a["kind"] == "spike" for a in result["alerts"]))

    def test_evaluate_unknown_rule_key_rejected(self) -> None:
        with self.assertRaises(NotFoundError):
            self.rig.anomaly.evaluate(
                INST_A, rule_key="no-such-key", values=[1, 2, 3])

    def test_get_unknown_evaluation_rejected(self) -> None:
        with self.assertRaises(NotFoundError):
            self.rig.anomaly.get_evaluation("aev-9999")

    def test_register_rule_requires_supervisor(self) -> None:
        with self.assertRaises(PermissionDeniedError):
            self.rig.anomaly.register_rule(INST_A, rule_key=RULE_KEY)

    def test_register_rule_validates_parameters(self) -> None:
        for overrides in (
            {"baseline_window": 1},
            {"baseline_window": True},
            {"sigma_threshold": 0},
            {"sigma_threshold": "3"},
            {"min_run": 0},
            {"rule_key": "  "},
        ):
            with self.assertRaises(ValidationError, msg=str(overrides)):
                self.register_rule(**overrides)

    def test_evaluate_validates_series(self) -> None:
        self.register_rule()
        for bad in ("not-a-list", [1, "x", 3], [True, 1, 2, 3, 4, 5]):
            with self.assertRaises(ValidationError, msg=repr(bad)):
                self.rig.anomaly.evaluate(
                    INST_A, rule_key=RULE_KEY, values=bad)


if __name__ == "__main__":
    unittest.main()

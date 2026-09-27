"""HTTP 边界：完整业务流、认证与错误映射。"""
from __future__ import annotations

import io
import json
import tempfile
import unittest

from service_09252_010.interfaces.wsgi_app import make_app

SUP = {"X-Institution-Id": "主管单位", "X-Role": "supervisor"}
INST_A = {"X-Institution-Id": "机构A"}


def call(app, method: str, path: str, body: dict | None = None,
         headers: dict | None = None, query: str = ""):
    payload = json.dumps(body).encode("utf-8") if body is not None else b""
    env = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "CONTENT_LENGTH": str(len(payload)),
        "wsgi.input": io.BytesIO(payload),
    }
    for key, value in (headers or {}).items():
        env["HTTP_" + key.upper().replace("-", "_")] = value
    captured: dict = {}

    def start_response(status, response_headers):
        captured["status"] = int(status.split()[0])

    chunks = app(env, start_response)
    return captured["status"], json.loads(b"".join(chunks).decode("utf-8"))


class HttpApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="svc09252-http-")
        self.app = make_app(f"{self._tmp.name}/api.db")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_full_acceptance_flow(self) -> None:
        # 指标登记（主管单位）
        status, body = call(self.app, "POST", "/indicators", {
            "code": "enrollment_total", "name": "招生总数", "category": "招生",
            "unit": "人", "formula": {"type": "sum",
                                      "measure": "enrollment_count"},
            "missing_policy": "skip",
        }, SUP)
        self.assertEqual(status, 201, body)

        # 机构授权：导入 + 计算 + 查看 + 导出
        for permission in ("import", "calculate", "view", "export"):
            status, _ = call(self.app, "POST", "/grants", {
                "institution_id": "机构A", "project_id": "P1",
                "category": "*", "permission": permission,
            }, SUP)
            self.assertEqual(status, 201)

        # 证据与数据导入（机构）
        status, body = call(self.app, "POST", "/evidence", {
            "project_id": "P1", "kind": "年报",
            "uri": "s3://ev/2024.pdf", "sha256": "a" * 64,
        }, INST_A)
        self.assertEqual(status, 201, body)
        evidence_id = body["evidence_id"]

        status, body = call(self.app, "POST", "/projects/P1/imports", {
            "reason": "首批",
            "records": [
                {"measure": "enrollment_count", "period": "2023-12",
                 "caliber": "DE-DUAL", "value": 10,
                 "evidence_id": evidence_id},
                {"measure": "enrollment_count", "period": "2024-01",
                 "caliber": "CN-STD", "value": 25,
                 "evidence_id": evidence_id},
            ],
        }, INST_A)
        self.assertEqual(status, 201, body)
        self.assertEqual(body["version_no"], 1)

        # 换算规则与会签
        status, body = call(self.app, "POST", "/rules", {
            "measure": "enrollment_count", "from_caliber": "DE-DUAL",
            "to_caliber": "CN-STD", "factor": 2.0,
        }, SUP)
        self.assertEqual(status, 201, body)
        rule_id = body["rule_id"]
        for signatory in ("supervisor", "partner_institution", "expert_panel"):
            status, body = call(self.app, "POST",
                                f"/rules/{rule_id}/signatures",
                                {"signatory": signatory}, SUP)
            self.assertEqual(status, 200, body)
        self.assertTrue(body["activated"])

        # 幂等计算：同一键提交两次，再执行
        task_body = {"project_id": "P1", "window_start": "2023-12",
                     "window_end": "2024-01", "target_caliber": "CN-STD",
                     "idempotency_key": "http-key-1"}
        status, first = call(self.app, "POST", "/tasks", task_body, INST_A)
        self.assertEqual(status, 201, first)
        status, second = call(self.app, "POST", "/tasks", task_body, INST_A)
        self.assertTrue(second["already_exists"])
        self.assertEqual(first["task_id"], second["task_id"])

        status, run = call(self.app, "POST",
                           f"/tasks/{first['task_id']}/run", {}, INST_A)
        self.assertEqual(status, 200, run)
        report_id = run["report_id"]

        status, report = call(self.app, "GET", f"/reports/{report_id}",
                              headers=INST_A)
        self.assertEqual(status, 200, report)
        self.assertEqual(report["lines"][0]["value"], 45.0)  # 10*2 + 25

        # 复核（主管单位）与导出（机构）
        status, _ = call(self.app, "POST", f"/reports/{report_id}/review",
                         {"approve": True}, SUP)
        self.assertEqual(status, 200)
        status, exported = call(self.app, "POST",
                                f"/reports/{report_id}/exports", {}, INST_A)
        self.assertEqual(status, 200, exported)
        self.assertEqual(exported["document"]["lines"][0]["value"], 45.0)

        # 复算核对
        status, check = call(self.app, "POST",
                             f"/reports/{report_id}/reverify", {}, SUP)
        self.assertEqual(status, 200, check)
        self.assertTrue(check["result_match"])

    def test_missing_institution_header_forbidden(self) -> None:
        status, body = call(self.app, "GET", "/indicators")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], "permission_denied")

    def test_unknown_route_404(self) -> None:
        status, _ = call(self.app, "GET", "/no-such", headers=SUP)
        self.assertEqual(status, 404)

    def test_validation_error_422(self) -> None:
        status, body = call(self.app, "POST", "/indicators", {
            "code": "x", "name": "x", "category": "招生",
            "formula": {"type": "unknown"},
        }, SUP)
        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "validation_error")

    def test_unauthorized_institution_403(self) -> None:
        status, body = call(self.app, "POST", "/projects/P1/imports", {
            "records": [{"measure": "m", "period": "2024-01",
                         "caliber": "CN-STD", "value": 1,
                         "evidence_id": "ev-1"}],
        }, {"X-Institution-Id": "陌生机构"})
        self.assertEqual(status, 403)

    def test_anomaly_monitoring_flow(self) -> None:
        # 机构无权登记监测规则
        status, body = call(self.app, "POST", "/anomaly-rules", {
            "rule_key": "P1|m|CN-STD",
        }, INST_A)
        self.assertEqual(status, 403)

        # 主管单位登记规则（基线窗口 4、阈值 3σ、连续 3 点判偏移）
        status, body = call(self.app, "POST", "/anomaly-rules", {
            "rule_key": "P1|m|CN-STD", "baseline_window": 4,
            "sigma_threshold": 3.0, "min_run": 3,
        }, SUP)
        self.assertEqual(status, 201, body)
        self.assertEqual(body["version_no"], 1)

        # 空序列：不误报
        status, body = call(self.app, "POST", "/anomaly-evaluations", {
            "rule_key": "P1|m|CN-STD", "values": [],
        }, INST_A)
        self.assertEqual(status, 201, body)
        self.assertEqual(body["verdict"], "normal")
        self.assertEqual(body["alerts"], [])

        # 一次尖峰与持续偏移的区分
        status, spike = call(self.app, "POST", "/anomaly-evaluations", {
            "rule_key": "P1|m|CN-STD",
            "values": [10, 11, 9, 10, 30, 10, 11],
        }, INST_A)
        self.assertEqual(status, 201, spike)
        self.assertEqual(spike["verdict"], "spike")
        self.assertEqual(spike["alerts"][0]["kind"], "spike")

        status, shift = call(self.app, "POST", "/anomaly-evaluations", {
            "rule_key": "P1|m|CN-STD",
            "values": [10, 11, 9, 10, 30, 29, 31, 10],
        }, INST_A)
        self.assertEqual(shift["verdict"], "shift")

        # 判定详情：原始序列与判定依据（规则快照）留痕
        status, detail = call(self.app, "GET",
                              f"/anomaly-evaluations/{spike['evaluation_id']}",
                              headers=INST_A)
        self.assertEqual(status, 200, detail)
        self.assertEqual(detail["rule_snapshot"]["sigma_threshold"], 3.0)
        self.assertEqual(detail["series"], [10.0, 11.0, 9.0, 10.0,
                                            30.0, 10.0, 11.0])

        # 规则变更后历史告警仍保留原判定依据
        status, body = call(self.app, "POST", "/anomaly-rules", {
            "rule_key": "P1|m|CN-STD", "baseline_window": 4,
            "sigma_threshold": 1000.0, "min_run": 3,
        }, SUP)
        self.assertEqual(body["version_no"], 2)
        status, detail = call(self.app, "GET",
                              f"/anomaly-evaluations/{spike['evaluation_id']}",
                              headers=INST_A)
        self.assertEqual(detail["rule_version_no"], 1)
        self.assertEqual(detail["verdict"], "spike")

        status, listing = call(self.app, "GET", "/anomaly-evaluations",
                               headers=SUP, query="rule_key=P1|m|CN-STD")
        self.assertEqual(status, 200, listing)
        self.assertEqual(len(listing["evaluations"]), 3)

        status, body = call(self.app, "GET",
                            "/anomaly-evaluations/aev-none", headers=SUP)
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()

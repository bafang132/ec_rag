"""
路线图新功能测试：Docker 无关（单元级），覆盖
- 多租户上下文与隔离
- 增量更新 / 文档版本管理
- 平台规则模板（Temu / Shopee / Lazada）
- 第三方评测对接（RAGAS / TruLens）优雅降级
- API Key 鉴权助手
"""
import pytest


def test_tenant_context_default_and_set():
    from app.shared.runtime.tenant import (
        DEFAULT_TENANT, get_tenant, reset_tenant, set_tenant,
    )

    reset_tenant()
    assert get_tenant() == DEFAULT_TENANT
    set_tenant("tenantA")
    assert get_tenant() == "tenantA"
    reset_tenant()
    assert get_tenant() == DEFAULT_TENANT


def test_version_service_roundtrip(tmp_path):
    import app.rag.import_.version_service as vs

    # 隔离持久化文件，避免污染项目 data/
    vs._LOCAL_PATH = tmp_path / "versions.json"

    f = tmp_path / "doc.md"
    f.write_text("hello world")
    h = vs.file_hash(f)
    assert len(h) == 64

    rec = vs.record_version("ItemX", h, 5, "src.md")
    assert rec["chunk_count"] == 5
    assert vs.get_latest_hash("ItemX") == h
    assert vs.is_unchanged("ItemX", h) is True
    assert vs.is_unchanged("ItemX", "different") is False

    vers = vs.list_versions("ItemX")
    assert len(vers) == 1 and vers[0]["chunk_count"] == 5

    # 重新记录同主体，历史累积且按租户隔离
    vs.record_version("ItemX", "hash2", 7, "src2.md")
    assert len(vs.list_versions("ItemX")) == 2
    assert vs.get_latest_hash("OtherItem") == ""


def test_platform_templates_coverage():
    from app.rag.import_ import platform_templates as pt

    for p in ("Temu", "Shopee", "Lazada"):
        tpl = pt.get_platform_template(p)
        assert tpl, f"{p} 应有规则模板"
        assert "highlights" in tpl and "risk_focus" in tpl
        assert pt.platform_rule_hint(p)  # 非空提示

    # 模糊匹配
    assert pt.get_platform_template("temu 半托管")
    # 未知道台返回 None
    assert pt.get_platform_template("不存在平台") is None


def test_third_party_graceful_when_missing():
    from evaluation import third_party as tp

    # available() 返回布尔；未安装时 evaluate 不应抛异常，返回结构化结果
    res = tp.evaluate([{"question": "q", "answer": "a", "context": "c"}])
    assert isinstance(res, dict)
    assert "available" in res
    if not tp.available():
        assert res["framework"] is None


def test_auth_helpers():
    from app.api.http.auth import DEFAULT_TENANT, _extract_key, _is_public

    class Req:
        def __init__(self, headers, path="/query"):
            self.headers = headers
            self.url = type("U", (), {"path": path})()

    # 公开路径放行
    assert _is_public("/health")
    assert _is_public("/html")
    assert _is_public("/admin/html")
    assert _is_public("/eval/html")
    assert not _is_public("/query")
    assert not _is_public("/api/eval/run")

    # key 提取
    assert _extract_key(Req({"authorization": "Bearer abc"})) == "abc"
    assert _extract_key(Req({"x-api-key": "xyz"})) == "xyz"
    assert _extract_key(Req({})) is None
    assert DEFAULT_TENANT == "default"


def test_apps_import_without_error():
    # 两个服务可正常导入（含鉴权中间件装配），且不因新功能报错
    from app.api.http import import_server, query_server

    assert import_server.app is not None
    assert query_server.app is not None
    # 新增路由存在
    paths = {r.path for r in query_server.app.routes}
    assert "/api/platform-templates" in paths
    ipaths = {r.path for r in import_server.app.routes}
    assert "/api/import/versions/{item_name}" in ipaths

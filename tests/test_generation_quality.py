"""
生成质量测试：意图识别（规则兜底）、合规风险检测、答案 Prompt 组装、三角色模板。

验证点：
1. 意图服务在 LLM 不可用时走规则兜底不中断；
2. 合规风险检测覆盖账号/知识产权/税务/禁售品/平台处罚五类；
3. 寒暄直达答案跳过检索；
4. 三角色模板正确切换，合规提醒注入上下文；
5. 答案结构化输出（引用来源）。
"""
import pytest

from app.shared.utils.ec_utils import (
    detect_compliance_risk,
    guess_category,
    guess_market,
    guess_platform,
    normalize_category,
    normalize_enum,
    normalize_platform,
    normalize_terms,
)


class TestTermNormalization:
    """术语归一化。"""

    def test_abbreviation_expansion(self):
        result = normalize_terms("FBA运费怎么算")
        assert "FBA 亚马逊物流" in result
        assert result.startswith("FBA运费怎么算")

    def test_no_hit_returns_original(self):
        text = "今天天气怎么样"
        assert normalize_terms(text) == text

    def test_max_terms_limit(self):
        result = normalize_terms("FBA VAT ACoS Listing CPC ROI", max_terms=2)
        assert result.count("；") <= 1  # 最多 2 个术语

    def test_word_boundary_english(self):
        # "ae" 不应命中 "速卖通别名 ae" 之外的普通单词
        result = normalize_terms("the cage is big")
        assert "速卖通" not in result


class TestComplianceRisk:
    """合规风险检测。"""

    def test_account_risk(self):
        risk = detect_compliance_risk("账号关联会封号吗")
        assert risk["risk_level"] in ("medium", "high")
        assert "账号封禁风险" in risk["risk_types"]

    def test_ip_risk(self):
        risk = detect_compliance_risk("产品被投诉侵权怎么办")
        assert "知识产权风险" in risk["risk_types"]

    def test_tax_risk(self):
        risk = detect_compliance_risk("VAT可以低申报避税吗")
        assert "税务合规风险" in risk["risk_types"]

    def test_multi_risk_is_high(self):
        risk = detect_compliance_risk("跟卖侵权被投诉会封号吗还要交VAT")
        assert risk["risk_level"] == "high"
        assert len(risk["risk_types"]) >= 2

    def test_no_risk(self):
        risk = detect_compliance_risk("FBA仓储费怎么计算")
        assert risk["risk_level"] == "none"
        assert risk["risk_warning"] == ""

    def test_empty_query(self):
        risk = detect_compliance_risk("")
        assert risk["risk_level"] == "none"


class TestEnumUtils:
    """枚举推断与规范化。"""

    def test_guess_platform(self):
        assert guess_platform("亚马逊FBA") == "亚马逊"
        assert guess_platform("TikTok Shop带货") == "TikTok Shop"
        assert guess_platform("虾皮店铺") == "Shopee"

    def test_guess_market(self):
        assert guess_market("欧洲VAT") == "欧洲"
        assert guess_market("日本站") == "日本"

    def test_guess_category(self):
        assert guess_category("ACoS优化") == "广告与推广"
        assert guess_category("退货政策") == "退换货与售后"
        assert guess_category("选品方法") == "选品与供应链"

    def test_normalize_platform_variants(self):
        assert normalize_platform("amazon") == "亚马逊"
        assert normalize_platform("亚马逊全球开店") == "亚马逊"
        assert normalize_platform("不认识的平台") == "未分类"

    def test_normalize_enum_containment(self):
        assert normalize_enum("物流仓储相关", ["物流仓储", "税务与财务"]) == "物流仓储"


class TestIntentService:
    """意图识别服务（规则兜底路径）。"""

    def test_greeting_skips_retrieval(self, isolated_history):
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.intent_service import confirm_ec_intent

        state = create_query_default_state(
            session_id="s1", original_query="你好", is_stream=False
        )
        result = confirm_ec_intent(state)
        assert result.get("answer")  # 直接产出答案
        assert result["ec_filters"]["needs_clarification"] is True

    def test_llm_failure_falls_back_to_rules(self, isolated_history, monkeypatch, stub_llm):
        """LLM 返回非 JSON 时走规则兜底，不中断链路。"""
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.intent_service import confirm_ec_intent

        # stub 返回无法解析的内容
        stub_llm.responses = ["这不是JSON，只是普通文本"]
        state = create_query_default_state(
            session_id="s2", original_query="亚马逊FBA仓储费怎么算", is_stream=False
        )
        result = confirm_ec_intent(state)
        # 兜底：改写问题 = 原问题，平台/类目由规则推断
        assert result["rewritten_query"] == "亚马逊FBA仓储费怎么算"
        assert result["ec_filters"]["platform"] == "亚马逊"
        assert result["ec_filters"]["category"] == "物流仓储"

    def test_compliance_risk_attached(self, isolated_history, monkeypatch, stub_llm):
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.intent_service import confirm_ec_intent

        stub_llm.responses = ['{"rewritten_query":"跟卖侵权投诉处理","platform":"亚马逊","market":"北美","category":"账号与合规","doc_type":"规则解读","needs_clarification":false}']
        state = create_query_default_state(
            session_id="s3", original_query="跟卖被投诉侵权会封号吗", is_stream=False
        )
        result = confirm_ec_intent(state)
        assert result["risk_level"] in ("medium", "high")
        assert result["risk_warning"]


class TestAnswerPrompt:
    """答案 Prompt 组装。"""

    def _docs(self):
        return [
            {"title": "FBA仓储规则", "file_title": "亚马逊FBA入仓与仓储规则.md",
             "content": "月度仓储费按立方英尺收取", "score": 0.85, "type": "local"},
        ]

    def test_role_template_selection(self, isolated_history):
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.answer_output_service import build_answer_prompt

        for role, expected in [
            ("seller", "【操作步骤】"),
            ("operator", "【执行方案】"),
            ("expert", "【合规审查】"),
        ]:
            state = create_query_default_state(
                session_id="s", original_query="q", role=role, is_stream=False
            )
            prompt = build_answer_prompt(self._docs(), "FBA仓储费", [], state)
            assert expected in prompt
            assert "月度仓储费" in prompt  # 上下文注入

    def test_compliance_warning_injected(self, isolated_history):
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.answer_output_service import build_answer_prompt

        state = create_query_default_state(
            session_id="s", original_query="q", role="seller", is_stream=False,
            risk_level="high", risk_warning="⚠️ 合规提醒：测试提醒内容",
        )
        prompt = build_answer_prompt(self._docs(), "q", [], state)
        assert "测试提醒内容" in prompt
        assert "【合规风险提醒】" in prompt

    def test_source_citation_in_context(self, isolated_history):
        from app.rag.query.answer_output_service import build_context_text

        context = build_context_text(self._docs())
        assert "《FBA仓储规则》" in context
        assert "匹配度" in context

    def test_empty_docs_degrades(self, isolated_history):
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.answer_output_service import build_answer_prompt

        state = create_query_default_state(session_id="s", original_query="q", is_stream=False)
        # 空检索不应抛异常
        prompt = build_answer_prompt([], "q", [], state)
        assert "q" in prompt

"""
跨境电商领域工具模块：术语归一化、合规风险检测、枚举规范化。

这些能力同时服务于：
- 查询链：改写查询时补全规范术语；检测合规风险并附加提醒；
- 导入链：元数据关键词兜底抽取。
"""
import re
from typing import Optional

from app.shared.config.ec_config import (
    CATEGORIES,
    COMPLIANCE_RISK_RULES,
    MARKETS,
    PLATFORMS,
    TERM_MAP,
    UNKNOWN_VALUE,
)
from app.shared.runtime.logger import logger


# ===================== 术语归一化 =====================

def normalize_terms(text: str, max_terms: int = 6) -> str:
    """
    将文本中的口语/缩写补全为跨境电商规范表述。

    策略：扫描 TERM_MAP 的 key（小写形式），命中则把该词及其规范表述
    一起拼接到文本尾部（而非原地替换），作为语义增强上下文，
    避免破坏原文语义的同时让向量检索能命中全称文档。

    :param text: 原始查询或文本
    :param max_terms: 最多补全的术语数，防止增强文本过长稀释语义
    :return: 补全规范术语后的文本
    """
    if not text:
        return text
    lowered = text.lower()
    hits: list[str] = []
    for term, canonical in TERM_MAP.items():
        if len(hits) >= max_terms:
            break
        # 英文术语按词边界匹配，中文术语按子串匹配
        if re.fullmatch(r"[a-z0-9+\- ]+", term):
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
            if re.search(pattern, lowered):
                hits.append(canonical)
        elif term in lowered:
            hits.append(canonical)
    if not hits:
        return text
    return f"{text} 相关术语：{'；'.join(hits)}"


# ===================== 平台/市场/类目识别（关键词兜底） =====================

PLATFORM_KEYWORDS: dict[str, list[str]] = {
    "亚马逊": ["亚马逊", "amazon", "fba", "fbm", "aws", "prime"],
    "速卖通": ["速卖通", "aliexpress", "ae"],
    "eBay": ["ebay", "e bay"],
    "Shopee": ["shopee", "虾皮"],
    "Lazada": ["lazada", "来赞达"],
    "TikTok Shop": ["tiktok", "tk shop", "抖音电商", "tiktok shop"],
    "Wish": ["wish"],
    "Temu": ["temu"],
    "SHEIN": ["shein"],
    "独立站": ["独立站", "shopify", "自建站", "shopify"],
}

MARKET_KEYWORDS: dict[str, list[str]] = {
    "北美": ["北美", "美国", "加拿大", "墨西哥", "us ", "usa", "america", "north america"],
    "欧洲": ["欧洲", "欧盟", "英国", "德国", "法国", "europe", "eu", "vat"],
    "日本": ["日本", "jp", "japan"],
    "东南亚": ["东南亚", "印尼", "泰国", "越南", "菲律宾", "马来", "新加坡"],
    "中东": ["中东", "沙特", "阿联酋", "迪拜"],
    "拉美": ["拉美", "巴西", "墨西哥"],
    "澳洲": ["澳洲", "澳大利亚", "新西兰"],
    "非洲": ["非洲", "尼日利亚", "肯尼亚"],
}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "物流仓储": ["物流", "fba", "fbm", "头程", "尾程", "仓储", "库存", "海运", "空运", "快递", "入库", "配送"],
    "退换货与售后": ["退货", "退款", "售后", "换货", "a-to-z", "索赔", "拒付", "退货率", "退货运费"],
    "Listing与商品运营": ["listing", "标题", "五点", "主图", "a+", "变体", "asin", "sku", "类目", "排名", "bsr", "a9", "跟卖", "购物车"],
    "广告与推广": ["广告", "cpc", "acos", "roas", "ppc", "竞价", "推广", "deal", "秒杀", "优惠券", "促销", "站外"],
    "账号与合规": ["账号", "二审", "kyc", "关联", "封号", "封店", "绩效", "侵权", "投诉", "注册", "审核"],
    "税务与财务": ["vat", "增值税", "税务", "关税", "税", "退税", "结汇", "汇率", "成本", "利润", "财务", "发票"],
    "支付与结算": ["支付", "收款", "回款", "手续费", "pingpong", "万里汇", "连连", "payoneer", "结算", "提现"],
    "选品与供应链": ["选品", "供应链", "供应商", "采购", "moq", "验货", "qc", "产品开发", "找货"],
    "平台规则": ["规则", "政策", "禁售", "违规", "处罚", "条款", "更新"],
    "营销活动": ["黑五", "网一", "prime day", "会员日", "大促", "活动", "旺季", "圣诞"],
}


def match_enum_by_keywords(text: str, rules: dict[str, list[str]], default: str = UNKNOWN_VALUE) -> str:
    """
    按关键词规则表匹配枚举值（加权词频计分）。

    计分方式：每个关键词贡献「出现次数 × 词长」分，取得分最高者。
    相比「最长单词优先」，词频计分更抗边缘提及干扰——
    例如文档通篇讲 FBA 仓储（fba×3 + 仓储×3 + 库存×2），
    仅在清货段落提到一次「站外Deal」时，仍能正确判定为「物流仓储」。

    :param text: 待匹配文本
    :param rules: 枚举值 -> 关键词列表
    :param default: 未命中时的兜底值
    :return: 命中的枚举值或兜底值
    """
    if not text:
        return default
    lowered = text.lower()
    best_value, best_score = default, 0
    for value, keywords in rules.items():
        score = 0
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if not kw_lower:
                continue
            occurrences = lowered.count(kw_lower)
            if occurrences:
                score += occurrences * len(kw_lower)
        if score > best_score:
            best_value, best_score = value, score
    return best_value


def guess_platform(text: str) -> str:
    """从文本推断平台（关键词兜底）。"""
    return match_enum_by_keywords(text, PLATFORM_KEYWORDS)


def guess_market(text: str) -> str:
    """从文本推断目标市场（关键词兜底）。"""
    return match_enum_by_keywords(text, MARKET_KEYWORDS)


def guess_category(text: str) -> str:
    """从文本推断业务类目（关键词兜底）。"""
    return match_enum_by_keywords(text, CATEGORY_KEYWORDS)


# ===================== 合规风险检测 =====================

def detect_compliance_risk(query: str) -> dict:
    """
    检测提问是否触及合规高风险领域。

    :param query: 用户原始问题
    :return: {"risk_level": none/medium/high, "risk_types": [...], "risk_warning": 合并后的提醒文案}
    """
    if not query:
        return {"risk_level": "none", "risk_types": [], "risk_warning": ""}
    lowered = query.lower()
    hit_types: list[str] = []
    warnings: list[str] = []
    for risk_type, (keywords, warning) in COMPLIANCE_RISK_RULES.items():
        for kw in keywords:
            if kw.lower() in lowered:
                hit_types.append(risk_type)
                warnings.append(warning)
                break
    if not hit_types:
        return {"risk_level": "none", "risk_types": [], "risk_warning": ""}
    # 命中 2 类及以上视为 high，否则 medium
    risk_level = "high" if len(hit_types) >= 2 else "medium"
    return {
        "risk_level": risk_level,
        "risk_types": hit_types,
        "risk_warning": "\n".join(warnings),
    }


# ===================== 枚举规范化 =====================

def normalize_enum(value: str, candidates: list[str], default: str = UNKNOWN_VALUE) -> str:
    """
    把任意取值归一化到受控枚举：先精确匹配，再包含匹配，最后取默认值。

    :param value: 原始取值（可能来自 LLM 输出或用户输入）
    :param candidates: 受控枚举列表
    :param default: 无法归一化时的兜底值
    :return: 归一化后的枚举值
    """
    if not value:
        return default
    value = str(value).strip()
    for candidate in candidates:
        if value == candidate:
            return candidate
    for candidate in candidates:
        if candidate in value:
            return candidate
    # 常见异形写法归一（大小写、连字符、别名）
    normalized = value.lower().replace("-", "").replace(" ", "").replace("店铺", "")
    for candidate in candidates:
        cand_norm = candidate.lower().replace("-", "").replace(" ", "")
        if cand_norm and (cand_norm in normalized or normalized in cand_norm):
            return candidate
    return default


def normalize_platform(value: str) -> str:
    """平台枚举规范化（无法归一化时按关键词推断，如 amazon→亚马逊）。"""
    normalized = normalize_enum(value, PLATFORMS)
    if normalized == UNKNOWN_VALUE:
        return guess_platform(value or "")
    return normalized


def normalize_market(value: str) -> str:
    """市场枚举规范化（无法归一化时按关键词推断，如 europe→欧洲）。"""
    normalized = normalize_enum(value, MARKETS)
    if normalized == UNKNOWN_VALUE:
        return guess_market(value or "")
    return normalized


def normalize_category(value: str) -> str:
    """类目枚举规范化（无法归一化时按关键词推断）。"""
    normalized = normalize_enum(value, CATEGORIES)
    if normalized == UNKNOWN_VALUE:
        return guess_category(value or "")
    return normalized


def safe_json_loads(text: str) -> Optional[dict]:
    """
    从模型输出中稳健提取 JSON 对象（兼容 ```json 围栏与前后多余文本）。

    :param text: 模型原始输出
    :return: 解析后的字典；失败返回 None
    """
    if not text:
        return None
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    import json

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        brace = re.search(r"\{.*\}", raw, re.DOTALL)
        if not brace:
            logger.warning(f"模型输出不是合法JSON，已忽略，原始输出：{text[:200]}")
            return None
        try:
            data = json.loads(brace.group(0))
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"模型输出无法解析为JSON，已忽略，原始输出：{text[:200]}")
            return None
    return data if isinstance(data, dict) else None

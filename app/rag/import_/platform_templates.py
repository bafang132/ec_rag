"""
平台规则模板模块：为各主流跨境电商平台提供结构化的「规则重点 / 合规关注 / 推荐文档类型」。

用途：
- 导入元数据抽取时，若识别到对应平台，自动附加 platform_rule_hint，强化该平台的合规模板；
- 前端导入表单可通过 /api/platform-templates 拉取，给出平台专属填写引导；
- 与 ec_config.PLATFORMS 对齐，新增平台只需在此补充模板即可。

路线图条目「更多平台规则模板（Temu / Shopee / Lazada）」的落地点。
"""
from typing import Optional

# 每个平台模板：
#   highlights   : 该平台卖家最关心的规则要点
#   risk_focus   : 该平台高频合规风险点（喂给合规风险雷达做强化）
#   doc_types    : 推荐上传的文档类型示例
#   market_hint  : 主要目标市场提示
PLATFORM_TEMPLATES: dict[str, dict] = {
    "亚马逊": {
        "market_hint": "北美 / 欧洲 / 日本",
        "highlights": [
            "FBA 入仓与仓储费（月度+长期仓储 LTSF）",
            "账号绩效（ODR/LSR）与二审/KYC",
            "A9 搜索与 Listing 合规（标题/变体/评论）",
            "广告 ACoS 与 Deal 提报规则",
        ],
        "risk_focus": ["账号封禁风险", "知识产权风险", "税务合规风险", "平台处罚风险"],
        "doc_types": ["平台政策", "规则解读", "实操教程"],
    },
    "Temu": {
        "market_hint": "北美 / 欧洲（全托管 + 半托管）",
        "highlights": [
            "全托管 / 半托管模式差异与定价机制",
            "质量审核（质检 QC）与罚款规则",
            "发货时效与履约考核（JIT / 备货）",
            "低价竞争与平台核价逻辑",
        ],
        "risk_focus": ["平台处罚风险", "禁售品风险", "知识产权风险"],
        "doc_types": ["平台政策", "实操教程", "规则解读"],
    },
    "Shopee": {
        "market_hint": "东南亚 / 拉美 / 台湾",
        "highlights": [
            "SLS 物流与藏价（运费补贴/运费模板）",
            "聊聊（Chat）客服响应率考核",
            "店铺评分与罚分（违反上架/刷单）",
            "活动报名与免运 / 虾币规则",
        ],
        "risk_focus": ["平台处罚风险", "禁售品风险", "知识产权风险"],
        "doc_types": ["平台政策", "实操教程", "营销活动"],
    },
    "Lazada": {
        "market_hint": "东南亚（六国）",
        "highlights": [
            "Lazada Grid / 跨境履约与税费",
            "店铺违规扣分与冻结规则",
            "Price Update / 价格与促销合规",
            "品牌与国际站点准入",
        ],
        "risk_focus": ["平台处罚风险", "禁售品风险", "税务合规风险"],
        "doc_types": ["平台政策", "实操教程", "规则解读"],
    },
    "eBay": {
        "market_hint": "北美 / 欧洲",
        "highlights": ["店铺评级（TSR）", "退款纠纷（INR/Return）", "刊登合规与 VeRO"],
        "risk_focus": ["知识产权风险", "平台处罚风险"],
        "doc_types": ["平台政策", "规则解读"],
    },
    "速卖通": {
        "market_hint": "全球 / 俄语区 / 巴西",
        "highlights": ["无忧物流与运费模板", "店铺分与违规处罚", "AE 活动与联盟"],
        "risk_focus": ["平台处罚风险", "禁售品风险"],
        "doc_types": ["平台政策", "实操教程"],
    },
    "TikTok Shop": {
        "market_hint": "东南亚 / 英美",
        "highlights": ["短视频/直播带货规则", "达人合作与联盟", "内容合规与禁售"],
        "risk_focus": ["禁售品风险", "平台处罚风险", "知识产权风险"],
        "doc_types": ["平台政策", "实操教程", "营销活动"],
    },
    "SHEIN": {
        "market_hint": "全球（平台招商模式）",
        "highlights": ["供应商备货与质检", "上新节奏与款式合规", "价格与回款"],
        "risk_focus": ["禁售品风险", "知识产权风险"],
        "doc_types": ["平台政策", "实操教程"],
    },
    "独立站": {
        "market_hint": "全球（Shopify 等）",
        "highlights": ["自建站合规（隐私/GDPR/支付）", "引流与广告政策", "税务与清关"],
        "risk_focus": ["税务合规风险", "禁售品风险"],
        "doc_types": ["规则解读", "实操教程"],
    },
}


def get_platform_template(platform: str) -> Optional[dict]:
    """按平台名返回规则模板（大小写/空白归一化匹配）。"""
    if not platform:
        return None
    key = platform.strip()
    if key in PLATFORM_TEMPLATES:
        return PLATFORM_TEMPLATES[key]
    # 模糊匹配：大小写不敏感的包含关系
    low = key.lower()
    for name, tpl in PLATFORM_TEMPLATES.items():
        low_name = name.lower()
        if low_name in low or low in low_name:
            return tpl
    return None


def platform_rule_hint(platform: str) -> str:
    """返回平台规则提示文案（用于元数据增强）。无匹配返回空串。"""
    tpl = get_platform_template(platform)
    if not tpl:
        return ""
    return f"{platform}规则重点：{'；'.join(tpl.get('highlights', [])[:3])}。"

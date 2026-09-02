"""The official 12-大类/84-细类 taxonomy from 特别需求补充.md §3.

特别需求补充.md's own header calls this "12 大类 / 83 细类", but the table it
defines (§3) actually lists 84 rows when counted directly — the discrepancy
is in the source document, not a transcription choice made here. This module
follows the table as written rather than silently dropping a row to match
the header count.

Both levels are modeled: 大类 (major category, for business aggregation and
UI display) and 细类 (subcategory, the finer-grained unit §3's table actually
enumerates, and the level §2's forced-manual-review list is defined at).
Both fast and advanced mode ask the model for a 细类 (with 大类 derived from
it) alongside the item's natural-language name/description, then snap
whatever the model returns to the closest official value here — never trust
a raw model string as a category label outright, matching the project's
overall don't-fabricate-don't-pass-through-unchecked posture.
"""
from __future__ import annotations

MAJOR_CATEGORIES: tuple[str, ...] = (
    "货币及涉案款",
    "账户及金融权益",
    "有价票证及支付凭证",
    "贵重物品",
    "电子设备及数字载体",
    "文件、证照及书证",
    "车辆及交通工具",
    "普通实物及作案工具",
    "枪支、弹药及管制器具",
    "毒品及违禁品",
    "危险及特殊保存物品",
    "特殊、大宗及新型财产",
)

FALLBACK_CATEGORY = "其他/未分类"

# 大类 whose member 细类 are wholly or mostly on the forced-manual-review
# list (特别需求补充.md §2) — used as a fail-safe fallback in
# needs_forced_review() when a 细类 could not be resolved, so an unresolved
# item inside one of these high-risk majors still gets flagged rather than
# silently passing as an ordinary result.
SENSITIVE_CATEGORIES: frozenset[str] = frozenset(
    {
        "枪支、弹药及管制器具",
        "毒品及违禁品",
        "危险及特殊保存物品",
    }
)

# Common variants/synonyms a model might produce instead of the exact
# official string — mapped to the canonical category so a near-miss isn't
# silently downgraded to FALLBACK_CATEGORY.
_ALIASES: dict[str, str] = {
    "货币": "货币及涉案款",
    "现金": "货币及涉案款",
    "金融账户": "账户及金融权益",
    "银行账户": "账户及金融权益",
    "票证": "有价票证及支付凭证",
    "支付凭证": "有价票证及支付凭证",
    "贵重物品及首饰": "贵重物品",
    "首饰": "贵重物品",
    "珠宝": "贵重物品",
    "电子产品": "电子设备及数字载体",
    "电子设备": "电子设备及数字载体",
    "数码产品": "电子设备及数字载体",
    "证件": "文件、证照及书证",
    "文件资料": "文件、证照及书证",
    "书证": "文件、证照及书证",
    "车辆": "车辆及交通工具",
    "交通工具": "车辆及交通工具",
    "日用品": "普通实物及作案工具",
    "生活用品": "普通实物及作案工具",
    "工具": "普通实物及作案工具",
    "衣物": "普通实物及作案工具",
    "服装": "普通实物及作案工具",
    "枪支弹药": "枪支、弹药及管制器具",
    "管制器具": "枪支、弹药及管制器具",
    "毒品": "毒品及违禁品",
    "违禁品": "毒品及违禁品",
    "危险品": "危险及特殊保存物品",
    "危化品": "危险及特殊保存物品",
    "特殊物品": "特殊、大宗及新型财产",
    "大宗物品": "特殊、大宗及新型财产",
}


def normalize_category(raw: str | None) -> str:
    """Snap a raw (possibly free-text) category guess to an official 大类.

    Exact match wins; then known aliases; then substring containment in
    either direction (handles the model appending/prepending extra words);
    anything else — including None/empty — becomes FALLBACK_CATEGORY rather
    than a fabricated-looking but wrong official category.
    """
    candidate = (raw or "").strip()
    if not candidate:
        return FALLBACK_CATEGORY
    if candidate in MAJOR_CATEGORIES:
        return candidate
    if candidate in _ALIASES:
        return _ALIASES[candidate]
    for official in MAJOR_CATEGORIES:
        if official in candidate or candidate in official:
            return official
    for alias, official in _ALIASES.items():
        if alias in candidate:
            return official
    return FALLBACK_CATEGORY


def is_sensitive(category: str) -> bool:
    return category in SENSITIVE_CATEGORIES


# ---------------------------------------------------------------------------
# 细类 (subcategory) — 特别需求补充.md §3's actual recognition-unit table.
# ---------------------------------------------------------------------------

SUBCATEGORIES: dict[str, tuple[str, ...]] = {
    "货币及涉案款": ("人民币现金", "外币"),
    "账户及金融权益": ("银行卡或储值卡", "存折"),
    "有价票证及支付凭证": ("支票或汇票", "有价证券凭证"),
    "贵重物品": (
        "黄金或贵金属",
        "名表",
        "珠宝玉石",
        "戒指",
        "项链",
        "手镯或手链",
        "耳环或耳饰",
        "其他首饰饰品",
        "收藏品",
    ),
    "电子设备及数字载体": (
        "手机",
        "笔记本电脑",
        "平板电脑",
        "台式电脑及配件",
        "硬盘",
        "U盘",
        "存储卡",
        "服务器或网络设备",
        "摄像设备",
        "光盘",
        "监控录像设备",
        "录音录像磁带",
        "其他电子设备",
    ),
    "文件、证照及书证": (
        "身份证件",
        "发票或票据",
        "印章",
        "账本或账册",
        "合同、证照及一般文书",
    ),
    "车辆及交通工具": (
        "汽车或机动车",
        "摩托车",
        "电动自行车",
        "自行车",
        "船舶",
        "车钥匙或遥控钥匙",
    ),
    "普通实物及作案工具": (
        "钱包",
        "背包",
        "手提包",
        "其他包袋",
        "上衣或衬衫",
        "裤子",
        "外套",
        "鞋",
        "胸罩",
        "内裤",
        "其他衣物",
        "刀具",
        "斧头",
        "剪刀",
        "锤子",
        "螺丝刀",
        "小型器具或机械",
        "日用品或普通商品",
    ),
    "枪支、弹药及管制器具": ("枪支", "枪械零部件", "弹药", "弩", "其他管制器具"),
    "毒品及违禁品": ("毒品", "毒品原植物或种子", "吸毒器具", "化学品容器（含危化品及制毒物品）", "淫秽物品"),
    "危险及特殊保存物品": ("爆炸物或雷管", "烟花爆竹", "液化气罐或易燃容器", "成品油", "放射性物品"),
    "特殊、大宗及新型财产": (
        "药品",
        "医疗器械",
        "食品或生鲜",
        "保健品或化妆品",
        "烟草制品",
        "活体动物",
        "野生动物及制品",
        "植物或植物制品",
        "大型机器设备或大宗货物",
        "房产或土地权属材料",
        "加密货币硬件钱包",
        "密钥或助记词载体",
    ),
}

SUBCATEGORY_TO_MAJOR: dict[str, str] = {
    subcategory: major for major, subcategories in SUBCATEGORIES.items() for subcategory in subcategories
}

# 特别需求补充.md §2: regardless of confidence, a result landing in one of
# these 细类 must force manual confirmation in the UI — never auto-pass.
FORCED_MANUAL_REVIEW_SUBCATEGORIES: frozenset[str] = frozenset(
    {
        "有价证券凭证",
        "收藏品",
        "监控录像设备",
        "化学品容器（含危化品及制毒物品）",
        "淫秽物品",
        "毒品",
        "成品油",
        "放射性物品",
        "野生动物及制品",
        "大型机器设备或大宗货物",
        "房产或土地权属材料",
        "密钥或助记词载体",
    }
)


def major_of_subcategory(subcategory: str) -> str | None:
    return SUBCATEGORY_TO_MAJOR.get(subcategory)


def normalize_subcategory(raw: str | None) -> str | None:
    """Snap free text to an official 细类; None if nothing resolves.

    Unlike normalize_category, there is no fallback bucket to snap to here —
    an unresolved 细类 must stay None (caller falls back to 大类-only
    classification) rather than being coerced into a wrong-but-official
    subcategory label.
    """
    candidate = (raw or "").strip()
    if not candidate:
        return None
    if candidate in SUBCATEGORY_TO_MAJOR:
        return candidate
    for official in SUBCATEGORY_TO_MAJOR:
        if official in candidate or candidate in official:
            return official
    return None


def is_forced_manual_review_subcategory(subcategory: str | None) -> bool:
    return subcategory in FORCED_MANUAL_REVIEW_SUBCATEGORIES if subcategory else False


def needs_forced_review(category: str, subcategory: str | None = None) -> bool:
    """特别需求补充.md §2's forced-manual-review rule.

    Applied at 细类 precision when a subcategory resolved (matching the
    doc's actual per-细类 wording — e.g. "收藏品" forces review but a
    sibling 细类 like "戒指" in the same 贵重物品 大类 does not). Falls back
    to the coarser 大类-level SENSITIVE_CATEGORIES check when no subcategory
    could be resolved, so an item that only got as far as "毒品及违禁品"
    still gets flagged rather than silently passing for lack of 细类 detail.
    """
    if subcategory:
        return is_forced_manual_review_subcategory(subcategory)
    return is_sensitive(category)


def format_taxonomy_block() -> str:
    """Render "[大类] 细类1、细类2、..." lines for embedding in a VLM prompt."""
    lines = []
    for major in MAJOR_CATEGORIES:
        subs = "、".join(SUBCATEGORIES[major])
        lines.append(f"   [{major}] {subs}")
    return "\n".join(lines)

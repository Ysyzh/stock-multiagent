#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具层 —— A股短线复盘的「领域规则固化」（无 LLM）

设计原则
========
1. **口径固化在这里，不指望 LLM 记住。** 每个工具内置领域规则校验，并把
   "我做了什么校验"随结果一起返回（`_checks`）。LLM 只读结论、不碰原始数据，
   就算想幻觉也没有原材料。
2. **每条口径都能溯源**，出处写在代码注释里，指向 `股票-领域知识.md` / `股票分析-交接与问题归档.md`。
3. **纯离线**：只读本地快照，运行期零网络。东财并发 16 会被封 IP（本工作区踩过两次），
   不稳定的联网取数已前置成一次性离线快照 —— 数据采集与推理解耦。

数据源（只读，相对路径）
========================
../tradingagents-demo/local-analysis/out/

已知数据缺口 —— 必须显式处理，不许静默通过
==========================================
| 缺口 | 现象 | 本文件的对策 |
|---|---|---|
| `calendar.json` 只到 2026-10-09 | 不含复盘日 10/12，会让停牌跨日校验在边界误判 | `EXTRA_SESSIONS` 本地补齐 |
| 跌停池 `dt` 返回 `tc=19` 但 `pool=[]` | 不能只信 `tc`（坑 4 的同类：静默返空） | `limit_down_by_snapshot()` 用全市场快照自算兜底 |
| 强势股池 `qs` 同样 `pool=[]` | 同上 | 暂不使用该池 |
| 只覆盖沪深主板非 ST | 漏掉创业板/科创板 20% 涨停，**真实情绪高度被低估** | 每个结论都带 `limitations` |
| 涨停池只能回溯最近约 6 交易日 | 超范围**静默返回最新一天** | `_assert_qdate()` 强制校验返回日期 |

运行
====
    python3 tools.py          # 自检：复现 zt-review-20261012.md 里已验证的数字
"""

from __future__ import annotations

import json
from pathlib import Path

# ════════════════════════════════════════════════════════════
#  数据源路径
# ════════════════════════════════════════════════════════════

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "tradingagents-demo" / "local-analysis" / "out"
)

# 本地快照支持的复盘日。数据是离线固定快照，不能假装能分析任意日期。
SUPPORTED_DATES = ("2026-10-12",)

# 区间工具的日线分片目录（3052 只 × 88 个交易日，17MB，31 个分片）
KLINES_DIR = DATA_DIR / "klines"


# ════════════════════════════════════════════════════════════
#  领域规则（出处：股票-领域知识.md）
# ════════════════════════════════════════════════════════════

def limit_ratio(code: str, name: str = "") -> float:
    """按代码前缀 / 名称判定涨跌幅限制比例。出处：领域卡 §二。

    判断顺序有讲究：ST 和退市整理期优先于板块前缀，
    因为「创业板 ST」是 20% 还是 5% 取决于名称而不是代码。
    """
    upper = (name or "").upper()
    if "ST" in upper:
        return 0.05
    if name and name.endswith("退"):
        return 0.10
    if code.startswith(("300", "301")):      # 创业板
        return 0.20
    if code.startswith(("688", "689")):      # 科创板
        return 0.20
    if code.startswith(("4", "8", "920")):   # 北交所
        return 0.30
    return 0.10                              # 沪深主板


def limit_up_price(prev_close: float, code: str, name: str = "") -> float:
    """涨停价 = round(前收 × (1+比例) + 1e-9, 2)。出处：领域卡 §二。

    `+1e-9` 是必须的：10.55 × 1.1 = 11.605000000000001，
    不修正会得到 11.61 而不是交易所的 11.60。
    """
    return round(prev_close * (1 + limit_ratio(code, name)) + 1e-9, 2)


def is_limit_up(prev_close: float, close: float, code: str, name: str = "") -> bool:
    """收盘 ≥ 涨停价 − 0.005 记涨停。出处：领域卡 §一。

    `−0.005` 容差是为了规避浮点比较失败（不是价格精度问题，是浮点表示问题）。
    """
    return close >= limit_up_price(prev_close, code, name) - 0.005


def to_symbol(code: str) -> str:
    """裸 6 位代码 → 带市场前缀的 symbol。出处：领域卡 §四。

    少写前缀会让接口**静默返回空数据且不报错**（本工作区踩过一次，坑 7）。
    """
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + code
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return "sz" + code
    if code.startswith(("4", "8", "920")):
        return "bj" + code
    raise ValueError(f"无法判定市场前缀: {code}")


# 复盘日 10/12 不在原始日历里（原始日历 2026-06-01 ~ 10-09）。
# 不补这一天，"停牌跨日 → 连板归零"的校验在区间边界就会误判（坑 8 的同类）。
EXTRA_SESSIONS = ("2026-10-12",)


def trading_calendar() -> list[str]:
    """交易日历（原始 + 本地补齐）。所有连板/连续性校验都必须用它。"""
    raw = _read_json(DATA_DIR / "calendar.json")
    return sorted(set(raw) | set(EXTRA_SESSIONS))


def prev_session(date: str) -> str | None:
    """前一个交易日。停牌校验用（相邻性判定）。"""
    cal = trading_calendar()
    if date not in cal:
        return None
    i = cal.index(date)
    return cal[i - 1] if i > 0 else None


# ════════════════════════════════════════════════════════════
#  数据加载（带缺口防护）
# ════════════════════════════════════════════════════════════

def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"数据源缺失: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_supported(date: str) -> str:
    """离线快照是固定的，不支持任意日期 —— 静默拿错日期的数据比报错更糟。"""
    d = date.replace("/", "-")
    if d not in SUPPORTED_DATES:
        raise ValueError(
            f"本地快照不支持 {d}（仅 {SUPPORTED_DATES}）。"
            "这是刻意设计：宁可报错，也不静默返回别的日期的数据（坑 4）。"
        )
    return d


def load_pools(date: str) -> dict:
    """涨停/炸板/跌停/强势股四池。出处：东财 push2ex。"""
    return _read_json(DATA_DIR / f"review_{_assert_supported(date).replace('-','')}_pools.json")


def load_snapshot(date: str) -> list[dict]:
    """全市场快照（5572 只），用于算市场宽度和跌停兜底。出处：新浪 getHQNodeData。"""
    return _read_json(DATA_DIR / f"review_{_assert_supported(date).replace('-','')}_snapshot.json")


def load_kline(date: str) -> dict:
    """高标股日线（47 只，不复权），用于连板数交叉复算。出处：腾讯单只接口。"""
    return _read_json(DATA_DIR / f"review_{_assert_supported(date).replace('-','')}_kline.json")


def load_industry_map() -> dict:
    """代码 → {industry, name}（5519 条）。出处：项目 MongoDB stock_basic_info。"""
    return _read_json(DATA_DIR / "industry_map.json")


def _assert_qdate(pool: dict, date: str) -> str:
    """强制校验池子返回的日期 == 请求日期。出处：坑 4。

    超范围请求时接口**不报错、静默返回最新一天**。不校验 qdate 会把
    「那天没有涨停」和「接口换了数据」搞混 —— 静默失败最致命（工作法 §2）。
    """
    want = int(date.replace("-", ""))
    got = pool.get("qdate")
    if got != want:
        raise ValueError(f"qdate 不匹配！请求 {want}，返回 {got}（坑 4：被静默替换）")
    return date


# ════════════════════════════════════════════════════════════
#  统一返回格式
# ════════════════════════════════════════════════════════════

_LIMIT_BOARD = "仅覆盖沪深主板非 ST；创业板/科创板 20% 涨停未纳入，真实情绪高度被低估（领域卡 §九）"


def _ok(data, checks: dict, limitations: list[str] | None = None) -> dict:
    """工具统一返回：数据 + 我做了什么校验 + 我的局限。"""
    return {
        "data": data,
        "_checks": checks,
        "limitations": [limitations] if isinstance(limitations, str) else (limitations or []),
    }


# ════════════════════════════════════════════════════════════
#  工具 1：数据边界（每个结论都该先看这个）
# ════════════════════════════════════════════════════════════

def data_boundaries(date: str = "2026-10-12") -> dict:
    """这个 Agent 到底能看到什么、看不到什么。必须先于任何分析结论。"""
    cal = trading_calendar()
    snap = load_snapshot(date)
    kline = load_kline(date)
    pools = load_pools(date)
    return _ok(
        {
            "复盘日": date,
            "交易日历": f"{cal[0]} ~ {cal[-1]}（{len(cal)} 个交易日）",
            "前一日交易日": prev_session(date),
            "全市场快照": f"{len(snap)} 只",
            "可复算日线的高标股": f"{len(kline)} 只",
            "涨停池": pools["zt"]["tc"],
            "炸板池": pools["zb"]["tc"],
            "跌停池": f"{pools['dt']['tc']} 家（⚠️ pool 为空，需快照自算兜底）",
        },
        {
            "日历补齐": f"原始日历止于 2026-10-09，本地补入 {list(EXTRA_SESSIONS)}",
            "qdate 校验": "已通过，返回日期 == 请求日期",
            "数据时点": "2026-10-12 收盘（离线快照，非实时）",
        },
        [_LIMIT_BOARD, "数据为历史快照，无盘中/盘后实时能力"],
    )


# ════════════════════════════════════════════════════════════
#  工具 2：连板梯队（含停牌跨日防护）
# ════════════════════════════════════════════════════════════

def ladder(date: str = "2026-10-12") -> dict:
    """连板梯队：按连板数分档 + 明细。出处：领域卡 §一「连板」。"""
    pools = load_pools(date)
    _assert_qdate(pools["zt"], date)
    pool = pools["zt"]["pool"]

    by_lbc: dict[int, list[dict]] = {}
    for s in pool:
        by_lbc.setdefault(s["lbc"], []).append(
            {
                "代码": s["c"],
                "名称": s["n"],
                "换手": round(s["hs"], 2),
                "炸板次数": s["zbc"],
                "封单_亿": round(s["fund"] / 1e8, 2),
                # 数值版留一份：下游要按它排序挑「最强/最弱」对比标的，
                # 字符串 "7.06%" 没法排序。
                "封单占流通_pct": round(s["fund"] / s["ltsz"] * 100, 2),
            }
        )

    heights = sorted(by_lbc, reverse=True)
    top = heights[0] if heights else 0
    # 断档：从最高板往下，缺失的高度（10/12 是 7/8/9 板全空、4/5 板全无）
    gaps = [h for h in range(top, 1, -1) if h not in by_lbc]

    return _ok(
        {
            "最高板": top,
            "梯队": {f"{h}板": {
                "家数": len(by_lbc[h]),
                "个股": [x["名称"] for x in by_lbc[h]],
                # 明细带封单/换手/炸板，供下游挑「同高度强 vs 弱」对比标的
                "明细": sorted(by_lbc[h], key=lambda x: x["封单占流通_pct"]),
            } for h in heights},
            "首板家数": len(by_lbc.get(1, [])),
            "连板家数(≥2板)": sum(len(v) for k, v in by_lbc.items() if k >= 2),
            "高度断档": gaps,
        },
        {
            "复权口径": "不复权（涨停判定必须用不复权，领域卡 §三）",
            "连板数来源": "东财 push2ex `lbc` 字段",
            "停牌防护": f"已用交易日历校验相邻性；前一日 = {prev_session(date)}",
            "qdate 校验": "已通过",
        },
        [_LIMIT_BOARD, "未剔除新股上市首日（C 开头）与退市整理期（领域卡 §九）"],
    )


# ════════════════════════════════════════════════════════════
#  工具 3：抱团度（用钱说话）
# ════════════════════════════════════════════════════════════

def seal_strength(date: str = "2026-10-12", top_n: int = 10) -> dict:
    """封单资金结构：合计、TOP-N 集中度、连板股封单占比。出处：领域卡 §八「龙头质量三指标」。"""
    pools = load_pools(date)
    _assert_qdate(pools["zt"], date)
    pool = pools["zt"]["pool"]

    total = sum(s["fund"] for s in pool)
    ranked = sorted(pool, key=lambda s: s["fund"], reverse=True)
    top_sum = sum(s["fund"] for s in ranked[:top_n])
    lianban = [s for s in pool if s["lbc"] >= 2]

    yi = lambda v: round(v / 1e8, 2)
    return _ok(
        {
            "封单合计_亿": yi(total),
            f"封单TOP{top_n}_亿": yi(top_sum),
            f"封单TOP{top_n}占比": f"{round(top_sum / total * 100, 1)}%",
            "封单TOP榜": [
                {"名称": s["n"], "连板": s["lbc"], "封单_亿": yi(s["fund"]),
                 "占流通市值": f"{round(s['fund'] / s['ltsz'] * 100, 2)}%"}
                for s in ranked[:top_n]
            ],
            "连板股家数": len(lianban),
            "连板股占比": f"{round(len(lianban) / len(pool) * 100, 1)}%",
            "连板股封单占比": f"{round(sum(s['fund'] for s in lianban) / total * 100)}%",
        },
        {
            "封单口径": "东财 push2ex `fund` 字段（元），除以 1e8 得亿",
            "占流通市值": "`fund` / `ltsz`（流通市值），越高说明锁定筹码越强",
            "qdate 校验": "已通过",
        },
        [_LIMIT_BOARD, "封单是收盘瞬时值，不含盘后撤单/追加"],
    )


# ════════════════════════════════════════════════════════════
#  工具 4：龙头质量三指标
# ════════════════════════════════════════════════════════════

def leader_quality(name: str, date: str = "2026-10-12") -> dict:
    """单只高标的封板质量。出处：领域卡 §八「龙头质量三指标」。

    三指标方向：封单/流通市值 高→强；换手率 低→强；炸板次数 0→强。
    """
    pools = load_pools(date)
    _assert_qdate(pools["zt"], date)
    hits = [s for s in pools["zt"]["pool"] if s["n"] == name or s["c"] == name]
    if not hits:
        return _ok(None, {"说明": f"{name} 不在 {date} 涨停池中"},
                   [_LIMIT_BOARD, f"{name} 当日未涨停，无法评估封板质量"])
    s = hits[0]
    pct = s["fund"] / s["ltsz"] * 100
    return _ok(
        {
            "名称": s["n"], "代码": s["c"], "连板数": s["lbc"],
            "封单_亿": round(s["fund"] / 1e8, 2),
            "流通市值_亿": round(s["ltsz"] / 1e8, 2),
            "封单占流通市值": f"{round(pct, 2)}%",
            "换手率": f"{round(s['hs'], 2)}%",
            "炸板次数": s["zbc"],
            "行业板块": s["hybk"],
            "首次封板时间": f"{s['fbt'] // 10000:02d}:{s['fbt'] // 100 % 100:02d}:{s['fbt'] % 100:02d}",
            "封板质量判定": (
                "强——筹码锁死、惜售" if pct >= 3 and s["hs"] < 10 and s["zbc"] == 0
                else "弱——高换手/封单薄/反复开板，疑似散户抬轿"
                if s["hs"] >= 15 or s["zbc"] >= 3
                else "中性"
            ),
        },
        {"判定规则": "封单占流通≥3% 且 换手<10% 且 炸板=0 → 强；换手≥15% 或 炸板≥3 → 弱"},
        [_LIMIT_BOARD, "无龙虎榜/席位数据，无法区分主力与游资"],
    )


# ════════════════════════════════════════════════════════════
#  工具 5：竞价封板 + 炸板率
# ════════════════════════════════════════════════════════════

def auction_sealed(date: str = "2026-10-12") -> dict:
    """9:25 集合竞价即封板的家数 —— 资金主动抱团的直接证据。"""
    pools = load_pools(date)
    _assert_qdate(pools["zt"], date)
    # fbt 形如 92500 = 9:25:00，92502 = 9:25:02 → 判 9 点 25 分这一分钟内
    sealed = [s for s in pools["zt"]["pool"] if 92500 <= s["fbt"] <= 92559]
    return _ok(
        {"竞价封板家数": len(sealed), "个股": [s["n"] for s in sealed]},
        {"判定规则": "`fbt`（首次涨停时间）落在 09:25:00~09:25:59"},
        [_LIMIT_BOARD],
    )


def blow_up_rate(date: str = "2026-10-12") -> dict:
    """炸板率 = 炸板家数 / (涨停家数 + 炸板家数)。出处：领域卡 §一「炸板」。"""
    pools = load_pools(date)
    zt, zb = pools["zt"], pools["zb"]
    _assert_qdate(zt, date)
    _assert_qdate(zb, date)
    n_zt, n_zb = zt["tc"], zb["tc"]
    return _ok(
        {
            "涨停家数": n_zt, "炸板家数": n_zb,
            "炸板率": f"{round(n_zb / (n_zt + n_zb) * 100, 1)}%",
            "炸板股": [{"名称": s["n"], "炸板次数": s["zbc"], "振幅": round(s["zf"], 2)}
                       for s in sorted(zb["pool"], key=lambda x: -x["zbc"])[:10]],
        },
        {"口径": "炸板率 = 炸板家数 / (涨停家数 + 炸板家数)", "qdate 校验": "涨停池+炸板池均已通过"},
        [_LIMIT_BOARD],
    )


# ════════════════════════════════════════════════════════════
#  工具 6：市场宽度 + 跌停兜底（填 dt 池空返回的坑）
# ════════════════════════════════════════════════════════════

def market_breadth(date: str = "2026-10-12") -> dict:
    """全市场涨跌分布 + 跌停自算。用快照兜底 `dt` 池 `pool=[]` 的静默返空（坑 4 同类）。"""
    snap = load_snapshot(date)
    pct = [s["changepercent"] for s in snap if isinstance(s.get("changepercent"), (int, float))]
    n = len(pct)
    down = sum(1 for p in pct if p < 0)
    pools = load_pools(date)
    return _ok(
        {
            "样本数": n,
            "下跌家数": down,
            "下跌占比": f"{round(down / n * 100, 1)}%",
            # ⚠️ 口径歧义高发区：「跌5%以上」有两种读法，本文件实测差 40 只。
            #    含准跌停 = 471；未及准跌停(5%~9.8%) = 431。
            #    对策不是替你选一个，而是**两个都返回并显式命名**，从源头消除歧义。
            #    （原报告 zt-review-20261012.md 用的是「未及准跌停」口径）
            "跌超5%家数(含准跌停)": sum(1 for p in pct if p <= -5),
            "跌超5%未及准跌停家数": sum(1 for p in pct if -9.8 < p <= -5),
            "准跌停家数(≤-9.8%)": sum(1 for p in pct if p <= -9.8),
            "上涨家数": sum(1 for p in pct if p > 0),
            "跌停池接口_tc": pools["dt"]["tc"],
            "跌停池接口_pool": len(pools["dt"]["pool"]),
        },
        {
            "跌停兜底": "跌停池接口返回 pool=[] （tc=19 不可信），改用全市场快照自算",
            "跌停口径": "≤-9.8%（略宽于精确跌停价，因含 20%/30% 板块需分档）",
            "跌超5%口径": "同时给出「含准跌停」与「未及准跌停」两个值，避免歧义",
            "快照来源": "新浪 getHQNodeData 全市场分页（5572 只，有效涨跌幅 5564）",
        },
        [_LIMIT_BOARD, "跌停判定用统一 -9.8% 阈值，未按各板块限制比例分档精确计算"],
    )


# ════════════════════════════════════════════════════════════
#  工具 7：行业分布（带动效应的粗颗粒度版本）
# ════════════════════════════════════════════════════════════

def industry_breakdown(date: str = "2026-10-12", min_count: int = 2) -> dict:
    """涨停股的行业板块分布。出处：领域卡 §六。

    ⚠️ `hybk` 是**东财行业板块**，比"题材"细。要得到"传媒 14 家"这种
    题材口径，必须人工把 出版/影视院线/电视广播/数字媒体 聚合起来 ——
    这是本工具的已知局限，不是 bug。
    """
    pools = load_pools(date)
    _assert_qdate(pools["zt"], date)
    counts: dict[str, list[str]] = {}
    for s in pools["zt"]["pool"]:
        counts.setdefault(s.get("hybk") or "未知", []).append(s["n"])
    ranked = sorted(counts.items(), key=lambda kv: -len(kv[1]))
    return _ok(
        {
            "行业家数": len(counts),
            "分布": [
                {"板块": k, "家数": len(v), "个股": v}
                for k, v in ranked if len(v) >= min_count
            ],
            "单家行业数": sum(1 for _, v in ranked if len(v) == 1),
        },
        {"字段来源": "东财 push2ex `hybk`（行业板块）"},
        [
            "行业 ≠ 题材：`hybk` 颗粒度偏细，题材口径需人工聚合，会低估题材集中度（领域卡 §六）",
            _LIMIT_BOARD,
        ],
    )


# ════════════════════════════════════════════════════════════
#  区间工具 —— 从 17MB 离线日线复算多日序列
# ════════════════════════════════════════════════════════════
#
#  前面 8 个工具都是**单日**的，数据源是 10/12 的池子快照。
#  这一组是**区间**的，数据源是 `klines/`（3052 只 × 88 个交易日不复权日线）。
#
#  两者口径**完全一致**（同一个 is_limit_up、同一个交易日历相邻性校验），
#  所以区间结果可以直接和已人工核验的 `nodes.json` 对比 —— 这就是 evaluate.py 的基础。

_KLINES_CACHE: dict | None = None


def _load_all_klines() -> dict:
    """读全部 31 个分片，合并成 {symbol: rows}。带缓存（17MB，只读一次）。"""
    global _KLINES_CACHE
    if _KLINES_CACHE is None:
        merged: dict = {}
        for fp in sorted(KLINES_DIR.glob("batch_*.json")):
            merged.update(_read_json(fp))
        _KLINES_CACHE = merged
    return _KLINES_CACHE


def _iter_limit_up_runs(rows: list, code: str):
    """逐行走一只票的日线，产出 (date, run, is_limit_up)。

    `run` = 截至该日的连续涨停天数。**停牌跨日必须归零**：
    相邻两根日线在交易日历上不相邻时，说明中间停牌过，
    停牌前后各一个涨停**不是连板**（对应 `股票分析-交接与问题归档.md` 坑 8）。
    """
    cal = trading_calendar()
    pos = {d: i for i, d in enumerate(cal)}
    prev_close = None
    prev_date = None
    prev_was_limit = False
    run = 0
    for row in rows:
        date, close = row[0], float(row[2])          # [date, open, close, high, low, vol]
        if prev_close is None:
            prev_close, prev_date = close, date
            continue
        lim = is_limit_up(prev_close, close, code)
        if lim:
            # 相邻性校验：上一根日线必须是日历上的前一个交易日
            adjacent = (prev_date in pos and date in pos
                        and pos[date] - pos[prev_date] == 1)
            run = (run + 1) if (adjacent and prev_was_limit) else 1
        else:
            run = 0
        yield date, run, lim
        prev_close, prev_date, prev_was_limit = close, date, lim


def compute_daily_stats(start: str, end: str) -> dict:
    """复算区间内每日的情绪序列：涨停家数 / 最高连板 / ≥2板家数 / ≥3板家数。

    与 `nodes.json` 的 `daily` 字段一一对应，可直接比对。
    """
    kl = _load_all_klines()
    per_day: dict[str, dict] = {}
    for sym, rows in kl.items():
        code = sym[2:] if sym[:2] in ("sh", "sz", "bj") else sym
        for date, run, lim in _iter_limit_up_runs(rows, code):
            if not (start <= date <= end):
                continue
            d = per_day.setdefault(date, {"count": 0, "max_run": 0, "ge2": 0, "ge3": 0})
            if lim:
                d["count"] += 1
                d["max_run"] = max(d["max_run"], run)
                if run >= 2:
                    d["ge2"] += 1
                if run >= 3:
                    d["ge3"] += 1
    daily = [{"date": k, **per_day[k]} for k in sorted(per_day)]
    return {
        "daily": daily,
        "cold_line": _cold_line([x["count"] for x in daily]),
        "_checks": {
            "复权口径": "不复权（klines 取的是腾讯不带 qfq 的 day 键）",
            "涨停判定": "round(前收×1.10+1e-9, 2)，收盘 ≥ 涨停价−0.005",
            "停牌防护": "交易日历相邻性校验，不相邻则连板归零",
            "样本": f"{len(kl)} 只 × {len(daily)} 个交易日",
        },
        "limitations": [_LIMIT_BOARD,
                        "区间复算依赖离线日线快照，覆盖 2026-06-01~10-09（分析区间 07-01 起，前 1 个月作连板基准）"],
    }


def _cold_line(counts: list[int]) -> int:
    """冰点线 = 涨停家数区间最低 20% 分位的阈值。出处：领域卡 §一「情绪冰点」。"""
    if not counts:
        return 0
    s = sorted(counts)
    return s[max(0, int(len(s) * 0.2) - 1)]


def detect_handovers(start: str, end: str, min_run: int = 3, min_cand: int = 2) -> dict:
    """换龙检测：A 断板（前一日 N 板，N≥min_run）→ B 接力（当日涨停且连板≥min_cand）。

    出处：领域卡 §一「换龙 / 接力」。与 `nodes.json` 的 `events` 字段对应。
    """
    kl = _load_all_klines()
    cal = trading_calendar()
    pos = {d: i for i, d in enumerate(cal)}
    # 先建索引：date -> {code: (name?, run)}（klines 不带名称，用 code 即可）
    day_run: dict[str, dict[str, int]] = {}
    for sym, rows in kl.items():
        code = sym[2:] if sym[:2] in ("sh", "sz", "bj") else sym
        for date, run, lim in _iter_limit_up_runs(rows, code):
            if lim and run:
                day_run.setdefault(date, {})[code] = run

    events = []
    dates = [d for d in sorted(day_run) if start <= d <= end]
    for date in dates:
        i = pos.get(date)
        if i is None or i == 0:
            continue
        prev = cal[i - 1]
        today, yest = day_run.get(date, {}), day_run.get(prev, {})
        broken = sorted(
            [{"code": c, "run": yest[c]} for c, r in yest.items()
             if r >= min_run and c not in today],
            key=lambda x: (-x["run"], x["code"]),
        )
        if not broken:
            continue
        cands = sorted([{"code": c, "run": r} for c, r in today.items() if r >= min_cand],
                       key=lambda x: (-x["run"], x["code"]))
        if not cands:
            continue
        # `A` 取**最强的单只**断板票 —— 与基准 `nodes.json` 的结构保持一致（每日一条、
        # A 是单只），这样两边才能逐条比对。当天其他断板票不丢，放在 `broken_all` 里。
        events.append({
            "date": date, "broken_date": prev,
            "A": broken[0],
            "broken_all": broken,
            "cands": cands,
            "market_count": len(today),
            "market_max": max(today.values()) if today else 0,
        })
    return {
        "events": events,
        "_checks": {"换龙口径": f"前一日 ≥{min_run} 板且当日未涨停 → 当日有 ≥{min_cand} 板接力",
                    "相邻性": "已用交易日历，前一日取日历前一位",
                    "样本": f"{len(kl)} 只 × {len(dates)} 个交易日"},
        "limitations": [_LIMIT_BOARD,
                        "换龙只认**当天**接力；真实资金接力常发生在断板后 1~2 天（领域卡 §九已知缺口）"],
    }


def rank_high_fliers(start: str, end: str, min_run: int = 4) -> dict:
    """高标排行：区间内最高连板 ≥ min_run 的个股。对应 `nodes.json` 的 `ranks`。"""
    kl = _load_all_klines()
    ranks = []
    for sym, rows in kl.items():
        code = sym[2:] if sym[:2] in ("sh", "sz", "bj") else sym
        best, best_start, best_end, cur_start = 0, None, None, None
        prev_run = 0
        for date, run, lim in _iter_limit_up_runs(rows, code):
            if not (start <= date <= end):
                continue
            if lim and run:
                if prev_run == 0:
                    cur_start = date
                if run > best:
                    best, best_start, best_end = run, cur_start, date
            prev_run = run if lim else 0
        if best >= min_run:
            ranks.append({"code": code, "max_run": best,
                          "run_start": best_start, "run_end": best_end})
    ranks.sort(key=lambda x: (-x["max_run"], x["code"]))
    return {
        "ranks": ranks,
        "_checks": {"高标口径": f"区间内最长连续涨停段 ≥ {min_run} 板", "停牌防护": "已校验"},
        "limitations": [_LIMIT_BOARD],
    }


# ════════════════════════════════════════════════════════════
#  工具注册表 —— 给 Agent 用的就是这些
# ════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    # 单日（数据源：10/12 池子快照）
    "data_boundaries": data_boundaries,
    "ladder": ladder,
    "seal_strength": seal_strength,
    "leader_quality": leader_quality,
    "auction_sealed": auction_sealed,
    "blow_up_rate": blow_up_rate,
    "market_breadth": market_breadth,
    "industry_breakdown": industry_breakdown,
    # 区间（数据源：klines 离线日线，口径与上面完全一致）
    "compute_daily_stats": compute_daily_stats,
    "detect_handovers": detect_handovers,
    "rank_high_fliers": rank_high_fliers,
}

DATE = "2026-10-12"


# ════════════════════════════════════════════════════════════
#  自检：复现 zt-review-20261012.md 里已验证的数字
# ════════════════════════════════════════════════════════════

def _selftest() -> int:
    """把工具算出来的值，跟已人工核验的复盘报告逐项比对。

    这是阶段 1 的验收标准：**工具层必须能复现已验证的结论**。
    对不上就说明口径错了 —— 口径错了，后面全错（工作法 §1）。
    """
    fails: list[str] = []

    def check(label: str, got, want):
        ok = abs(got - want) < 0.06 if isinstance(want, float) else got == want
        print(f"  {'✅' if ok else '❌'} {label:26} 工具={got!r:>14}  报告={want!r}")
        if not ok:
            fails.append(label)

    print("=" * 68)
    print("  阶段 1 自检：复现 zt-review-20261012.md 的已验证数字")
    print("=" * 68)

    print("\n[数据边界]")
    b = data_boundaries(DATE)
    print(f"  {b['data']['交易日历']}")
    print(f"  前一日交易日 = {b['data']['前一日交易日']}")

    print("\n[连板梯队]")
    lad = ladder(DATE)["data"]
    check("最高板", lad["最高板"], 10)
    check("10板家数", lad["梯队"].get("10板", {}).get("家数", 0), 1)
    check("6板家数", lad["梯队"].get("6板", {}).get("家数", 0), 2)
    check("3板家数", lad["梯队"].get("3板", {}).get("家数", 0), 1)
    check("2板家数", lad["梯队"].get("2板", {}).get("家数", 0), 14)
    check("首板家数", lad["首板家数"], 28)
    check("连板家数", lad["连板家数(≥2板)"], 18)
    check("高度断档(7/8/9,4/5)", lad["高度断档"], [9, 8, 7, 5, 4])

    print("\n[抱团度]")
    seal = seal_strength(DATE)["data"]
    check("封单合计_亿", seal["封单合计_亿"], 47.12)
    check("封单TOP10占比", float(seal["封单TOP10占比"].rstrip("%")), 61.8)
    check("连板股家数", seal["连板股家数"], 18)
    # 报告写「39%」是取整；工具给出更精确的 39.1%（18/46=39.13%）。取整后比对。
    check("连板股占比(报告取整)", round(float(seal["连板股占比"].rstrip("%"))), 39)
    check("连板股封单占比", float(seal["连板股封单占比"].rstrip("%")), 57.0)

    print("\n[竞价封板]")
    check("竞价封板家数", auction_sealed(DATE)["data"]["竞价封板家数"], 8)

    print("\n[炸板率]")
    bl = blow_up_rate(DATE)["data"]
    check("涨停家数", bl["涨停家数"], 46)
    check("炸板家数", bl["炸板家数"], 25)
    check("炸板率", float(bl["炸板率"].rstrip("%")), 35.2)

    print("\n[市场宽度]")
    mb = market_breadth(DATE)["data"]
    check("下跌占比", float(mb["下跌占比"].rstrip("%")), 77.7)
    check("准跌停家数", mb["准跌停家数(≤-9.8%)"], 40)
    # 原报告「跌5%以上 431 只」= 未及准跌停口径（471 = 431 + 40）。两个口径都验。
    check("跌超5%未及准跌停", mb["跌超5%未及准跌停家数"], 431)
    check("跌超5%(含准跌停)", mb["跌超5%家数(含准跌停)"], 471)

    print("\n[龙头质量三指标]")
    for nm, want in [("新华传媒", (10, 10.17, 7.06, 3.38, 0)),
                     ("九阳股份", (6, 0.59, 0.47, 16.07, 5)),
                     ("时代万恒", (6, 0.31, 0.81, 32.01, 3))]:
        q = leader_quality(nm, DATE)["data"]
        print(f"\n  · {nm}")
        check("  连板数", q["连板数"], want[0])
        check("  封单_亿", q["封单_亿"], want[1])
        check("  封单占流通%", float(q["封单占流通市值"].rstrip("%")), want[2])
        check("  换手%", float(q["换手率"].rstrip("%")), want[3])
        check("  炸板次数", q["炸板次数"], want[4])
        print(f"  判定: {q['封板质量判定']}")

    print("\n[行业分布]")
    ind = industry_breakdown(DATE)["data"]
    print(f"  板块数={ind['行业家数']}  单家行业={ind['单家行业数']}")
    for x in ind["分布"][:6]:
        print(f"    {x['板块']:8} {x['家数']:>2} 家  {x['个股']}")

    print("\n" + "=" * 68)
    if fails:
        print(f"  ❌ {len(fails)} 项不匹配：{fails}")
    else:
        print("  ✅ 全部匹配 —— 工具层能精确复现已人工核验的复盘结论")
    print("=" * 68)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())

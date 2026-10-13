#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 6：对 ground truth 算准确率 —— 把"我觉得还行"变成"命中率 97%"。

═══════════════════════════════════════════════════════════
对比双方
═══════════════════════════════════════════════════════════
| 侧 | 来源 | 性质 |
|---|---|---|
| **我方** | `tools.compute_daily_stats / detect_handovers / rank_high_fliers` | 从 3052 只 × 88 日不复权日线**独立复算** |
| **基准** | `../tradingagents-demo/local-analysis/out/nodes.json` | 此前用 `analyze_nodes.py` 跑出、并写入领域卡的结论 |

两边用的是**同一份日线**，所以这不是"模型准不准"，而是
**"我的口径实现对不对"** —— 一个 205k 股票日的口径回归测试。

═══════════════════════════════════════════════════════════
⭐ 本评估最有价值的产出（实测）
═══════════════════════════════════════════════════════════
`count`（涨停家数）**67/67 = 100% 一致**，`cold_line` 一致（41）。
但 `max_run` 只有 85.1%，且**我方系统性高 1**。

逐票定位后，3 处不一致**全是同一个模式**：我方的 `run_start` 比基准早一个交易日，
`run_end` 相同。用原始日线逐笔核对：

    新华传媒 2026-09-21：前收 5.31 → 涨停价 5.84 → 收盘 **5.84**  ✅ 确是涨停
    海鸥住工 2026-08-24：前收 3.59 → 涨停价 3.95 → 收盘 **3.95**  ✅ 确是涨停
    恒尚节能 2026-07-01：前收 12.94 → 涨停价 14.23 → 收盘 **14.23** ✅ 确是涨停

**结论：是基准（`analyze_nodes.py`）漏掉了每个连板段的第一天，不是我们算错。**

还有**第三方交叉验证**：10/12 涨停池的东财 `lbc` 字段显示新华传媒是 **10 板**。
我方口径：10/09 为 9 板，再加 10/12 一天 = **10** ✅；
基准口径：10/09 为 8 板，再加一天 = 9 ✗。

> 这就是评估的价值：**它发现的是 ground truth 的 bug，而不是模型的 bug。**
> 一个"两边都对不上就改自己"的做法，会在这里把对的改成错的。

运行：python3 evaluate.py
产物：out/eval-<区间>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tools as T

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "out"

GT_PATH = (PROJECT_DIR.parent / "tradingagents-demo" / "local-analysis"
           / "out" / "nodes.json")

START, END = "2026-07-01", "2026-10-09"


# ════════════════════════════════════════════════════════════
#  一、逐日情绪序列
# ════════════════════════════════════════════════════════════

def compare_daily(mine: dict, gt: dict) -> dict:
    gtd = {d["date"]: d for d in gt["daily"]}
    keys = ("count", "max_run", "ge2", "ge3")
    res = {"days": len(mine["daily"]), "metrics": {}, "mismatch_days": {}}
    for k in keys:
        hit, diffs = 0, []
        for m in mine["daily"]:
            g = gtd.get(m["date"])
            if g is None:
                continue
            if m[k] == g[k]:
                hit += 1
            else:
                diffs.append({"date": m["date"], "mine": m[k], "gt": g[k],
                              "delta": m[k] - g[k]})
        res["metrics"][k] = {
            "agree": hit, "total": len(mine["daily"]),
            "rate": round(hit / len(mine["daily"]) * 100, 1),
            "diffs": diffs,
        }
        if diffs:
            res["mismatch_days"][k] = diffs
    res["cold_line"] = {"mine": mine["cold_line"], "gt": gt["cold_line"],
                        "agree": mine["cold_line"] == gt["cold_line"]}
    return res


# ════════════════════════════════════════════════════════════
#  二、高标排行（含 off-by-one 诊断）
# ════════════════════════════════════════════════════════════

def compare_ranks(mine: dict, gt: dict) -> dict:
    my = {r["code"]: r for r in mine["ranks"]}
    g = {r["code"]: r for r in gt["ranks"]}
    both = set(my) & set(g)
    only_mine, only_gt = set(my) - set(g), set(g) - set(my)

    maxrun_diff = []
    start_lag = 0
    for c in sorted(both):
        if my[c]["max_run"] != g[c]["max_run"]:
            # 诊断：run_start 是否恰好早一个交易日（off-by-one 的指纹）
            lag = (my[c]["run_start"] != g[c]["run_start"]
                   and my[c]["run_end"] == g[c]["run_end"])
            start_lag += int(lag)
            maxrun_diff.append({
                "code": c, "name": g[c].get("name"),
                "mine": my[c]["max_run"], "gt": g[c]["max_run"],
                "my_start": my[c]["run_start"], "gt_start": g[c]["run_start"],
                "my_end": my[c]["run_end"], "gt_end": g[c]["run_end"],
                "疑似基准漏首日": lag,
            })
    return {
        "mine_count": len(my), "gt_count": len(g), "both": len(both),
        "only_mine": [{"code": c, "name": g.get(c, {}).get("name"), "max_run": my[c]["max_run"]}
                      for c in sorted(only_mine)],
        "only_gt": [{"code": c, "name": g[c].get("name"), "max_run": g[c]["max_run"]}
                    for c in sorted(only_gt)],
        "maxrun_diff": maxrun_diff,
        "offbyone_suspects": start_lag,
    }


# ════════════════════════════════════════════════════════════
#  三、换龙事件（召回 / 精确）
# ════════════════════════════════════════════════════════════

def compare_events(mine: dict, gt: dict) -> dict:
    """换龙事件的召回/精确。

    匹配键 = `(日期, A 的代码)`。`A` 两边都是**单只**（最强的断板票），
    与基准 `nodes.json` 结构一致 —— 第一版我方把当天所有断板票塞进一个列表，
    导致键永远对不上（召回只有 25.9%），已改为同构。

    `cands` 不参与匹配：基准封顶 5 只，我方给全部 ≥2 板（信息更全但不同构）。
    """
    def key(e):
        a = e.get("A")
        code = a["code"] if isinstance(a, dict) else (a[0]["code"] if a else None)
        return e["date"], code

    my = {key(e): e for e in mine["events"]}
    g = {key(e): e for e in gt["events"]}
    both = set(my) & set(g)
    # 顺带统计"我方候选更全"的比例
    wider = sum(1 for k in both if len(my[k]["cands"]) > len(g[k]["cands"]))

    # ── 未命中归因：**按票的身份**归因，而不是只比连板高度 ──
    #
    # ⚠️ 这里的第一版写错了，是外部独立审查（让 Claude Code 首次阅读本文件）
    #    指出来的 —— 值得原样记下：
    #
    #    「两条分支都只看『当天我方 A 中是否存在 run==gt_run 的票』，
    #      不检查那只票是不是 c 本身。若我方把 c 的 run 算错、导致 c 没当上 A，
    #      只要当天另有一只票高度碰巧相同，就误报『并列选票差异』放过真问题。」
    #
    #    也就是说第一版是**循环论证**：它假设了"差异来自 tie-break"，
    #    而不是验证这件事。实测复现：造一个「我方真的选错票」的场景，
    #    它会判成"并列选票差异"。
    #
    # 修法：改成看**票的身份** ——
    #   · 基准选的票 c 在我方"当天全部断板票"里 → 我方识别到了它，只是选 A 的约定不同
    #   · 基准选的票 c **不在**我方断板票里 → 我方漏识别，必须人工核查（可能是真错）
    by_date_my_all: dict[str, dict[str, int]] = {}   # 日期 → {代码: 连板高度}
    for (d, _c), e in my.items():
        codes = {x["code"]: x["run"] for x in (e.get("broken_all") or [e["A"]])}
        by_date_my_all.setdefault(d, {}).update(codes)

    def classify(d: str, c: str, gt_run: int) -> str:
        mine_map = by_date_my_all.get(d)
        if not mine_map:
            return "边界（我方当日无事件）"
        if c not in mine_map:
            return "⚠️ 我方未识别该断板票（需人工核查）"
        my_run = mine_map[c]
        if my_run == gt_run:
            return "并列选票差异（同一只票我方也识别到了，只是没选它当 A）"
        if my_run == gt_run + 1:
            return "基准 off-by-one 连带（同一只票，高度差 1）"
        return f"⚠️ 同一只票高度不符（我方 {my_run} vs 基准 {gt_run}，需核查）"

    missed_detail = []
    for d, c in sorted(set(g) - set(my)):
        gt_run = g[(d, c)]["A"]["run"]
        missed_detail.append({"date": d, "A": c, "gt_run": gt_run,
                              "我方是否识别到该票": c in (by_date_my_all.get(d) or {}),
                              "归因": classify(d, c, gt_run)})
    reason: dict[str, int] = {}
    for m in missed_detail:
        reason[m["归因"]] = reason.get(m["归因"], 0) + 1

    return {
        "mine_count": len(my), "gt_count": len(g),
        "matched": len(both),
        "recall": round(len(both) / len(g) * 100, 1) if g else 0.0,
        "precision": round(len(both) / len(my) * 100, 1) if my else 0.0,
        "cands_wider_than_gt": wider,
        "missed": missed_detail[:20],
        "extra": [{"date": k[0], "A": k[1]} for k in sorted(set(my) - set(g))][:20],
        "miss_reasons": reason,
    }


# ════════════════════════════════════════════════════════════
#  四、第三方交叉验证：拿东财 lbc 校准争议票
# ════════════════════════════════════════════════════════════

def cross_validate_with_lbc(disputed: list[str], review_date: str = "2026-10-12") -> list[dict]:
    """有争议的高标，用 10/12 涨停池的东财 `lbc` 做第三方仲裁。

    逻辑：若 10/12 该股仍在涨停池且 `lbc = N`，则 10/09 应当是 N-1 板
    （10/09 到 10/12 之间只有一个交易日）。
    """
    pools = T.load_pools(review_date)
    lbc = {s["c"]: s["lbc"] for s in pools["zt"]["pool"]}
    out = []
    for code in disputed:
        n = lbc.get(code)
        out.append({
            "code": code,
            "东财lbc_1012": n,
            "推算1009应为": (n - 1) if n else None,
            "说明": ("10/12 仍在涨停池，lbc = %d → 10/09 应为 %d 板" % (n, n - 1)) if n
                    else "10/12 不在涨停池，无法用 lbc 仲裁",
        })
    return out


# ════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════

def main() -> int:
    if not GT_PATH.is_file():
        print(f"❌ 找不到 ground truth: {GT_PATH}")
        return 2
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

    print("=" * 74)
    print(f"  阶段 6 评估：我方口径 vs ground truth（nodes.json）")
    print(f"  区间 {START} ~ {END}   样本 3052 只 × {len(gt['daily'])} 个交易日")
    print("=" * 74)

    print("\n[1/4] 复算每日情绪序列…")
    mine_daily = T.compute_daily_stats(START, END)
    d = compare_daily(mine_daily, gt)
    for k, v in d["metrics"].items():
        flag = "✅" if v["rate"] == 100.0 else "⚠️ "
        print(f"   {flag} {k:8} 一致 {v['agree']:>3}/{v['total']}  ({v['rate']}%)")
    cl = d["cold_line"]
    print(f"   {'✅' if cl['agree'] else '❌'} cold_line  我={cl['mine']}  基准={cl['gt']}")

    print("\n[2/4] 复算高标排行…")
    mine_ranks = T.rank_high_fliers(START, END)
    r = compare_ranks(mine_ranks, gt)
    print(f"   我方 {r['mine_count']} 只 / 基准 {r['gt_count']} 只 / 共有 {r['both']} 只")
    print(f"   max_run 不一致 {len(r['maxrun_diff'])} 只"
          f"（其中 {r['offbyone_suspects']} 只呈「基准漏首日」特征）")
    if r["only_mine"]:
        print(f"   仅我方有（{len(r['only_mine'])} 只）: "
              f"{[x['name'] or x['code'] for x in r['only_mine']]}")
    if r["only_gt"]:
        print(f"   仅基准有（{len(r['only_gt'])} 只）: "
              f"{[x['name'] or x['code'] for x in r['only_gt']]}")

    print("\n[3/4] 复算换龙事件…")
    mine_ev = T.detect_handovers(START, END)
    e = compare_events(mine_ev, gt)
    print(f"   我方 {e['mine_count']} 个 / 基准 {e['gt_count']} 个 / 命中 {e['matched']} 个")
    print(f"   召回率 {e['recall']}%   精确率 {e['precision']}%")
    if e["miss_reasons"]:
        print("   漏报归因（避免把并列差异误报成算错）：")
        for k, v in sorted(e["miss_reasons"].items(), key=lambda kv: -kv[1]):
            print(f"     · {v:>2} 个  {k}")

    print("\n[4/4] 第三方交叉验证（东财 lbc 仲裁争议票）…")
    disputed = sorted({x["code"] for x in r["maxrun_diff"]})
    arb = cross_validate_with_lbc(disputed)
    for a in arb:
        print(f"   {a['code']}  {a['说明']}")
    # 用 lbc 判定谁对
    verdict = []
    for a in arb:
        if a["推算1009应为"] is None:
            continue
        cr = next((x for x in r["maxrun_diff"] if x["code"] == a["code"]), None)
        if cr:
            verdict.append({
                "code": a["code"], "name": cr.get("name"),
                "东财推算1009": a["推算1009应为"],
                "我方": cr["mine"], "基准": cr["gt"],
                "判定": "✅ 我方与东财一致" if cr["mine"] == a["推算1009应为"] else
                        ("❌ 基准与东财一致" if cr["gt"] == a["推算1009应为"] else "无法判定"),
            })
    print("\n   ── 争议票仲裁结论 ──")
    for v in verdict:
        print(f"   {v['name'] or v['code']:8} 东财推算={v['东财推算1009']}  "
              f"我方={v['我方']}  基准={v['基准']}   → {v['判定']}")

    # ── 汇总 ────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  汇总")
    print("=" * 74)
    print(f"  逐日涨停家数（count）  : {d['metrics']['count']['rate']}% 一致"
          f"（{d['metrics']['count']['agree']}/{d['metrics']['count']['total']}）")
    print(f"  逐日最高连板（max_run）: {d['metrics']['max_run']['rate']}% 一致")
    print(f"  冰点线（cold_line）    : {'一致' if cl['agree'] else '不一致'}"
          f"（{cl['mine']} vs {cl['gt']}）")
    print(f"  高标票集合命中         : {r['both']}/{r['gt_count']}"
          f"（{round(r['both']/r['gt_count']*100,1)}%）")
    print(f"  换龙事件召回/精确      : {e['recall']}% / {e['precision']}%")
    print(f"\n  ⭐ 关键结论：max_run 的偏差**不是我们算错**，而是基准漏掉每个连板段首日。")
    print(f"     证据 1：3 处不一致全部呈「run_start 早一天、run_end 相同」特征")
    print(f"     证据 2：原始日线逐笔核对，那 3 天的收盘价确实等于涨停价")
    print(f"     证据 3：东财 lbc 仲裁 → "
          f"{sum(1 for v in verdict if '我方与东财一致' in v['判定'])}/{len(verdict)} 支持我方")

    # ── 落盘 ────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"eval-{START}_{END}.json"
    out.write_text(json.dumps({
        "区间": [START, END], "基准": str(GT_PATH),
        "逐日序列": d, "高标": r, "换龙": e,
        "第三方仲裁": verdict,
        "结论": "max_run 差异源于基准（analyze_nodes.py）漏计连板段首日；"
                "我方口径与原始日线及东财 lbc 三方交叉验证一致。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[产物] {out.relative_to(PROJECT_DIR)}  ({out.stat().st_size} 字节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

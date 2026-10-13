#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证两个关键行为：

  1. **并行扇出 + reducer 合并**：4 个分析师并行写 `findings`，
     必须全部保留（若 reducer 缺失会互相覆盖，最后只剩 1 份）。
  2. **reflection 回路真的会打回重写**，而不是只在 happy path 上跑通。

═══════════════════════════════════════════════════════════
为什么需要这个测试
═══════════════════════════════════════════════════════════
- 第一次实跑时审查员一次就放行了初稿 —— 也就是说 `rewrite` 这条边
  **从未被执行过**。没走过的分支不算验证过。
- 4 个分析师并行写同一字段，是 reducer 唯一"不可省"的场景；
  少了它不会报错，只会**静默丢数据** —— 这种 bug 必须用测试兜住。

做法：把 4 个分析师 + Supervisor + Writer 全换成假的（零 token），
     只有 `critique` 用**真的 LLM**。所以这一跑只花 2 次 critic 调用的 token。

═══════════════════════════════════════════════════════════
植入的缺陷（全部来自真实实测）
═══════════════════════════════════════════════════════════
| 代号 | 缺陷 | 植入方式 |
|---|---|---|
| ② | 自相矛盾 | 写「有 5 家是连板股」，括号里却列出 6 只 |
| ③ | 比较方向反 | 「东百 3.21 亿仅次于赛力斯 3.1 亿」（其实东百更大） |
| ④ | 计数错 | 「这 4 个高度」（7/8/9 只有 3 个） |
| D | 漏免责声明 | 全文不含「不构成任何投资建议」 |
| F | 局限不全 | 只列 2 条，实际有 7 条 |

运行：python3 test_reflection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.types import Command

import graph as G
from prompts import build_writer_prompt

# ── 故意有缺陷的初稿 ────────────────────────────────────────
BROKEN_V1 = """## 一、一句话结论

**这是一份用于测试的、故意植入缺陷的报告。**

## 二、连板梯队

| 高度 | 家数 |
|---|---|
| 10板 | 1 |
| 9板 | 0（断档） |
| 8板 | 0（断档） |
| 7板 | 0（断档） |
| 6板 | 2 |

- 竞价封板 8 家，其中**有 5 家是连板股**（新华传媒、九阳股份、华北制药、新经典、丽人丽妆、德才股份，实为 6 家）。

## 三、抱团证据

- 东百集团 3.21 亿在首板中**仅次于**赛力斯 3.1 亿。
- 10 板（1 家）到 6 板（2 家）之间没有任何 7/8/9 板，说明**这 4 个高度**上没有一只个股成功晋级。

## 八、口径与局限

1. 仅覆盖沪深主板非 ST。
2. 数据为历史快照。
"""

FIXED_V2 = """## 一、一句话结论

**测试用第 2 版：缺陷已修。**

## 二、连板梯队

- 竞价封板 8 家，其中**有 6 家是连板股**（新华传媒、九阳股份、华北制药、新经典、丽人丽妆、德才股份），另 2 家为首板（东百集团、广电网络）。

## 三、抱团证据

- 东百集团 3.21 亿，**高于**赛力斯的 3.1 亿（3.21 > 3.1）。
- 10 板到 6 板之间没有任何 7/8/9 板，说明**这 3 个高度**上没有个股成功晋级。

## 八、口径与局限

1. 仅覆盖沪深主板非 ST；创业板/科创板 20% 涨停未纳入。
2. 数据为历史快照，无盘中/盘后实时能力。
3. 未剔除新股上市首日与退市整理期。
4. 封单是收盘瞬时值。
5. 跌停判定用统一 -9.8% 阈值。
6. 行业 ≠ 题材。
7. 无龙虎榜/席位数据。

---

**免责声明**：本报告仅为历史数据描述，不构成任何投资建议。
"""

captured_prompts: list[str] = []


# ── 假节点：4 个分析师（零 token）────────────────────────────

def fake_analyst_factory(key: str, spec: dict):
    def node(state: G.ReplayState) -> dict:
        return {
            "findings": [{
                "analyst": key,
                "label": spec["label"],
                "focus": spec["focus"],
                "slices_used": [s for s in spec["slices"] if s in state["facts"]],
                "text": f"- 【{spec['label']}】这是一条假的视角结论（测试用，零 token）。",
            }],
            "usages": [],
            "trace": [{"step": f"analyst:{key}", "elapsed_s": 0.0, "llm_calls": 0}],
        }
    node.__name__ = f"analyst_{key}"
    return node


def fake_supervisor(state: G.ReplayState) -> dict:
    n = len(state.get("findings") or [])
    return {
        "digest": f"冲突点：假汇总（收到 {n} 份视角结论）。\n优先级：略。\n缺口：略。",
        "usages": [],
        "trace": [{"step": "supervisor", "elapsed_s": 0.0, "views_merged": n, "llm_calls": 0}],
    }


# ── 假节点：Writer（第 1 版坏、第 2 版好）────────────────────

def fake_writer(state: G.ReplayState) -> dict:
    version = state.get("revision", 0) + 1
    prior = state.get("critique") or []

    # 记录第 2 版收到的完整 prompt —— 这是"审查意见被注入"的证据
    captured_prompts.append(
        build_writer_prompt(state["date"], state["facts"], state["checks"],
                            state["limitations"], findings=state.get("findings"),
                            digest=state.get("digest", ""), critique=prior or None)
    )

    report = BROKEN_V1 if version == 1 else FIXED_V2
    return {
        "report": report,
        "revision": version,
        "usages": [],                       # 假装零成本
        "trace": [{
            "step": f"writer#{version}（fake{'·重写' if prior else ''}）",
            "elapsed_s": 0.0,
            "is_rewrite": bool(prior),
            "llm_calls": 0,
        }],
    }


def main() -> int:
    print("=" * 72)
    print("  多 Agent 回路测试：并行扇出 + reducer 合并 + reflection 打回")
    print("=" * 72)

    # 必须全部在 build_graph() 之前替换 —— add_node 会绑定当时的函数对象
    G._make_analyst_node = fake_analyst_factory       # 4 个分析师
    G.supervisor = fake_supervisor                    # 主管
    G.analyze = fake_writer                           # 执笔人
    app = G.build_graph()

    print("\n[图结构]")
    for line in app.get_graph().draw_mermaid().splitlines():
        if "-->" in line or "-." in line:
            print("   " + line.strip())
    print(f"   节点数 = {len([n for n in app.get_graph().nodes if not n.startswith('__')])}")

    final: dict = {}
    # 图现在带 checkpointer + 人工确认关，所以必须给 thread_id，
    # 并在遇到 interrupt 时送回答复（本测试自动通过人工关，专测自动回路）。
    config = {"configurable": {"thread_id": "t-reflection"}}
    payload = {"date": "2026-10-12"}
    for _ in range(6):
        interrupted = None
        for chunk in app.stream(payload, config=config, stream_mode="values"):
            final = chunk
            if chunk.get("__interrupt__"):
                interrupted = chunk["__interrupt__"]
        if not interrupted:
            break
        payload = Command(resume="approve")
    else:
        raise RuntimeError("轮次超限")

    print("\n[执行轨迹]")
    for s in final.get("trace", []):
        print(f"   {s['step']:26} llm={s.get('llm_calls')}")

    findings = final.get("findings", [])
    print(f"\n[视角结论] 收到 {len(findings)} 份："
          f"{[f.get('label') for f in findings]}")

    critique = final.get("critique", [])
    print(f"\n[审查回路] {len(critique)} 轮，最终版本 v{final.get('revision')}")
    for i, c in enumerate(critique, 1):
        print(f"   第{i}轮 ok={c.get('ok')} degraded={c.get('degraded', False)}")
        print(f"     总评：{c.get('summary', '')[:100]}")
        for f in c.get("findings", []):
            print(f"     · [{f.get('kind')}] {f.get('detail', '')[:88]}")

    # ── 断言 ────────────────────────────────────────────────
    print("\n[断言]")
    checks: list[tuple[str, bool]] = []

    # 1) 并行扇出 + reducer：4 份视角结论必须全部保留
    checks.append((f"4 份视角结论全部保留（reducer 生效，实收 {len(findings)}）",
                   len(findings) == len(G.ROLE_SPECS)))
    checks.append(("视角标签各不相同（无覆盖）",
                   len({f.get("analyst") for f in findings}) == len(G.ROLE_SPECS)))
    checks.append(("每个视角都拿到了切片",
                   all(f.get("slices_used") for f in findings)))
    checks.append(("主管汇总收到了全部 4 份",
                   "4 份视角结论" in (final.get("digest") or "")))

    # 2) 回路
    checks.append(("审查员判定初稿不通过", bool(critique) and critique[0].get("ok") is False))
    checks.append(("触发了重写（生成到 v2）", final.get("revision") == 2))
    checks.append(("共两轮审查", len(critique) == 2))
    checks.append(("第 1 轮有 findings", bool(critique and critique[0].get("findings"))))
    checks.append(("第 2 版已生成且是修复稿", "缺陷已修" in (final.get("report") or "")))

    # 3) 重写时只回到执笔人，不重跑分析师
    n_analyst_steps = sum(1 for s in final.get("trace", [])
                          if str(s.get("step", "")).startswith("analyst:"))
    checks.append((f"分析师只跑了一轮（{n_analyst_steps} 次，未随重写重跑）",
                   n_analyst_steps == len(G.ROLE_SPECS)))

    # 4) 熔断（关键安全属性）
    checks.append((f"熔断生效（revision 达上限 {G.MAX_REVISIONS} 后收工）",
                   final.get("revision") == G.MAX_REVISIONS))
    checks.append((f"审查轮数未失控（≤ {G.MAX_REVISIONS} 轮）",
                   len(critique) <= G.MAX_REVISIONS))

    # 5) 审查意见注入 + 审查质量
    inj = captured_prompts[-1] if len(captured_prompts) >= 2 else ""
    checks.append(("第 2 版 prompt 含「本轮是「重写」」标记", "本轮是「重写」" in inj))
    checks.append(("第 2 版 prompt 含第 1 轮审查意见", inj.count("修改方向") >= 1))
    # 正确检查是"4 个视角标签都出现"，而不是数某个短语的次数
    _labels = [sp["label"] for sp in G.ROLE_SPECS.values()]
    checks.append((f"第 2 版 prompt 含全部 {len(_labels)} 份视角结论",
                   all(lb in inj for lb in _labels)))
    allfix = " ".join(f.get("kind", "") + f.get("detail", "")
                      for f in (critique[0].get("findings") if critique else []))
    checks.append(("抓到「漏免责声明」红线", "免责" in allfix))
    checks.append(("抓到计数类问题", "计数" in allfix))
    checks.append(("未降级（审查真的生效）",
                   bool(critique) and not critique[0].get("degraded", False)))

    all_ok = True
    for label, good in checks:
        all_ok &= bool(good)
        print(f"   {'✅' if good else '❌'} {label}")

    print("\n" + "=" * 72)
    print("  ✅ 全部通过：4 视角并行产出 → reducer 合并 → 汇总 → 执笔 → 审查 → 打回重写 → 熔断收工"
          if all_ok else "  ❌ 有断言失败，见上")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

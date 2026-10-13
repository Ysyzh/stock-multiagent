#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 reflection 回路**真的会打回重写**，而不是只在 happy path 上跑通。

═══════════════════════════════════════════════════════════
为什么需要这个测试
═══════════════════════════════════════════════════════════
第一次实跑时，审查员一次就放行了初稿 —— 也就是说 `critique → analyze`
这条 `rewrite` 边**从未被执行过**。没走过的分支不算验证过。

做法：把 `analyze` 换成假的（零 token），但 `critique` 用**真的 LLM**：
  · 第 1 版故意植入我们实测过的三类错误 + 删掉免责声明
  · 第 2 版记录它收到的 prompt，用来证明**审查意见确实被注入**了

所以这一跑只花 2 次 critic 调用的 token。

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

import graph as G

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

- 东百集团 3.21 亿，**高于**赛力斯的 3.1 亿（=3.21 vs 3.1）。
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


def fake_analyze(state: G.ReplayState) -> dict:
    """假的 analyst：第 1 版给坏报告，第 2 版记录 prompt 后给好报告。"""
    version = state.get("revision", 0) + 1
    prior = state.get("critique") or []

    # 记录第 2 版收到的完整 prompt —— 这是"审查意见被注入"的证据
    from prompts import build_user_prompt
    captured_prompts.append(
        build_user_prompt(state["date"], state["facts"], state["checks"],
                          state["limitations"], critique=prior or None)
    )

    report = BROKEN_V1 if version == 1 else FIXED_V2
    return {
        "report": report,
        "revision": version,
        "usages": [],                       # 假装零成本，测试不关心
        "trace": [{
            "step": f"analyze#{version}（fake{'·重写' if prior else ''}）",
            "elapsed_s": 0.0,
            "is_rewrite": bool(prior),
            "llm_calls": 0,
        }],
    }


def main() -> int:
    print("=" * 70)
    print("  reflection 回路测试：故意植入缺陷，看审查员会不会打回")
    print("=" * 70)

    # 必须在 build_graph() 之前替换，因为 add_node 会绑定当时的函数对象
    G.analyze = fake_analyze
    app = G.build_graph()

    print("\n[图结构]")
    for line in app.get_graph().draw_mermaid().splitlines():
        if "-->" in line or "-." in line:
            print("   " + line.strip())

    final: dict = {}
    for chunk in app.stream({"date": "2026-10-12"}, stream_mode="values"):
        final = chunk

    print("\n[执行轨迹]")
    for s in final.get("trace", []):
        print(f"   {s['step']:26} rewrite={s.get('is_rewrite')}")

    critique = final.get("critique", [])
    print(f"\n[审查回路] {len(critique)} 轮，最终版本 v{final.get('revision')}")
    for i, c in enumerate(critique, 1):
        print(f"   第{i}轮 ok={c.get('ok')} degraded={c.get('degraded', False)}")
        print(f"     总评：{c.get('summary', '')[:100]}")
        for f in c.get("findings", []):
            print(f"     · [{f.get('kind')}] {f.get('location','')[:34]}")
            print(f"       {f.get('detail','')[:92]}")

    # ── 断言 ────────────────────────────────────────────────
    print("\n[断言]")
    checks: list[tuple[str, bool]] = []

    checks.append(("审查员判定初稿不通过", critique and critique[0].get("ok") is False))
    checks.append(("触发了重写（生成到 v2）", final.get("revision") == 2))
    checks.append(("共两轮审查", len(critique) == 2))
    checks.append(("第 1 轮有 findings", bool(critique and critique[0].get("findings"))))

    # ⚠️ 这里断言的是「第 2 版**确实被生成**」，而不是「第 2 版被审查通过」。
    #    因为本测试的第 2 版只是个骨架假报告，审查员严格拒收是**正确行为**。
    #    （上次我把断言写成检查输入 prompt 是否含修复稿内容，属于测试自身的 bug。）
    checks.append(("第 2 版已生成且是修复稿", "缺陷已修" in (final.get("report") or "")))

    # 关键安全属性：审查员连续不放行时，必须靠熔断收工，绝不能无限重写
    checks.append((f"熔断生效（revision 达上限 {G.MAX_REVISIONS} 后收工）",
                   final.get("revision") == G.MAX_REVISIONS))
    checks.append((f"审查轮数未失控（≤ {G.MAX_REVISIONS} 轮）",
                   len(critique) <= G.MAX_REVISIONS))

    # 审查意见是否真的注入了第 2 版的 prompt
    inj = captured_prompts[-1] if len(captured_prompts) >= 2 else ""
    kinds = " ".join(f.get("kind", "") for f in (critique[0].get("findings") if critique else []))
    checks.append(("第 2 版 prompt 含「本轮是「重写」」标记", "本轮是「重写」" in inj))
    checks.append(("第 2 版 prompt 含第 1 轮审查意见", inj.count("修改方向") >= 1))

    # 审查员是否抓到植入的缺陷类型
    for label, kw in [("抓到「漏免责声明」", "免责"), ("抓到数字/计数类问题", "计数")]:
        checks.append((f"{label}（findings 含『{kw}』）", kw in kinds or kw in
                       " ".join(f.get("detail", "") for f in (critique[0].get("findings") if critique else []))))
    checks.append(("未降级（审查真的生效）",
                   bool(critique) and not critique[0].get("degraded", False)))

    all_ok = True
    for label, good in checks:
        all_ok &= bool(good)
        print(f"   {'✅' if good else '❌'} {label}")

    print("\n" + "=" * 70)
    print("  ✅ 回路验证通过：打回 → 注入意见 → 重写 → 复审 → 收工"
          if all_ok else "  ❌ 有断言失败，见上")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

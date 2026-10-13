#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 **人工确认关（interrupt 暂停/恢复）** 的控制流 —— 零 token 成本。

═══════════════════════════════════════════════════════════
为什么要单独测这件事
═══════════════════════════════════════════════════════════
`interrupt()` 是**唯一一个会"暂停整个图"**的能力，它依赖三件事同时正确：
  1. `compile(checkpointer=...)` —— 没有 checkpointer 状态无处保存，无法恢复
  2. 恢复时必须用**同一个 thread_id** —— 否则找不到那份状态
  3. 恢复要用 `Command(resume=...)`，节点内的 `interrupt()` 才会返回答复

这三条任意一条错，表现都是"卡住"或"静默重跑"，**很难从日志里看出来**。
所以用假节点把 LLM 全部替换掉，只验控制流 —— 又快又免费。

覆盖两条路径：
  · **approve**：人工通过 → 直接收工
  · **reject:<意见>**：人工打回 → 回到执笔人按人的意见重写 → 再过审查 → 再问一次
    （并验证"人的意见确实被注入到重写的 prompt 里"）

运行：python3 test_interrupt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph.types import Command

import graph as G
from prompts import build_writer_prompt

writer_prompts: list[str] = []
critique_calls = [0]
# 假审查的剧本，每个场景自己重置：
#   True  = 审查通过；False = 审查不通过（逼出一次自动重写）
CRITIQUE_PLAN: list[bool] = [True]


# ── 假节点：全部零 LLM ──────────────────────────────────────

def fake_analyst_factory(key: str, spec: dict):
    def node(state):
        return {
            "findings": [{"analyst": key, "label": spec["label"],
                          "slices_used": list(spec["slices"]),
                          "text": f"- 【{spec['label']}】假结论"}],
            "usages": [], "trace": [{"step": f"analyst:{key}", "llm_calls": 0}],
        }
    node.__name__ = f"analyst_{key}"
    return node


def fake_supervisor(state):
    return {"digest": "假汇总：无冲突。", "usages": [],
            "trace": [{"step": "supervisor", "llm_calls": 0}]}


def fake_writer(state):
    writer_prompts.append(build_writer_prompt(
        state["date"], state["facts"], state["checks"], state["limitations"],
        findings=state.get("findings"), digest=state.get("digest", ""),
        critique=state.get("critique") or None,
        human_notes=state.get("human_notes") or None,
    ))
    rev = state.get("revision", 0) + 1
    return {
        "report": f"## 假报告 v{rev}\n\n内容略。",
        "revision": rev,
        "usages": [], "trace": [{"step": f"writer#{rev}", "llm_calls": 0}],
    }


def fake_critique(state):
    i = critique_calls[0]
    critique_calls[0] += 1
    ok = CRITIQUE_PLAN[i] if i < len(CRITIQUE_PLAN) else True
    return {
        "critique": [{"ok": ok,
                      "summary": "假审查：通过" if ok else "假审查：有问题",
                      "findings": [] if ok else [{"kind": "计数错误", "location": "第二节",
                                                  "detail": "假的", "fix_hint": "改掉"}],
                      "reviewed_version": state.get("revision", 1)}],
        "usages": [], "trace": [{"step": f"critique#{i+1}", "ok": ok, "llm_calls": 0}],
    }


def drive(app, thread_id: str, answers: list[str], max_rounds: int = 8):
    """跑图直到结束，遇到 interrupt 就用 answers 里的答复恢复。"""
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"date": "2026-10-12"}
    quests: list[dict] = []
    final: dict = {}
    for _ in range(max_rounds):
        interrupted = None
        for chunk in app.stream(payload, config=config, stream_mode="values"):
            final = chunk
            if chunk.get("__interrupt__"):
                interrupted = chunk["__interrupt__"]
        if not interrupted:
            return final, quests
        item = interrupted[0]
        quests.append(getattr(item, "value", item) or {})
        payload = Command(resume=answers.pop(0) if answers else "approve")
    raise RuntimeError("轮次超限：可能存在人工确认死循环")


def main() -> int:
    print("=" * 74)
    print("  人工确认关（interrupt）控制流测试 —— 全部假节点，零 token")
    print("=" * 74)
    checks: list[tuple[str, bool]] = []

    # ── 场景 1：审查通过 → 人工 approve → 收工 ──────────────
    print("\n【场景 1】审查一次通过 → 人工 approve")
    G._make_analyst_node = fake_analyst_factory
    G.supervisor, G.analyze, G.critique = fake_supervisor, fake_writer, fake_critique
    CRITIQUE_PLAN[:] = [True]              # 剧本：一次通过
    critique_calls[0] = 0
    writer_prompts.clear()
    app = G.build_graph()
    final, quests = drive(app, "t-approve", ["approve"])

    checks.append(("图里含 human_review 节点", "human_review" in app.get_graph().nodes))
    checks.append(("interrupt 确实触发了（图暂停过）", len(quests) == 1))
    checks.append(("待审批 payload 带报告版本", bool(quests) and quests[0].get("revision") == 1))
    checks.append(("待审批 payload 带报告全文", bool(quests and quests[0].get("report"))))
    checks.append(("待审批 payload 带审查结论", bool(quests and quests[0].get("critic_summary"))))
    checks.append(("approve 后 human_decision=approve",
                   (final.get("human_decision") or "").startswith("approve")))
    checks.append(("approve 后收工（未再重写，revision=1）", final.get("revision") == 1))
    print(f"   暂停 {len(quests)} 次；最终 revision={final.get('revision')}；"
          f"human_decision={final.get('human_decision')}")

    # ── 场景 2：自动审查先打回一次 + 人工 reject 一次 ────────
    print("\n【场景 2】自动审查打回一次 → 人工 reject → 再重写 → approve")
    CRITIQUE_PLAN[:] = [False, True, True, True]   # 剧本：第 1 次不通过，之后通过
    critique_calls[0] = 0
    writer_prompts.clear()
    app2 = G.build_graph()
    final2, quests2 = drive(app2, "t-reject",
                            ["reject:请在第八节补充数据时点说明", "approve"])

    traces = [s.get("step", "") for s in final2.get("trace", [])]
    n_writer = sum(1 for s in traces if s.startswith("writer#"))
    n_analyst = sum(1 for s in traces if s.startswith("analyst:"))
    inj = writer_prompts[-1] if writer_prompts else ""

    print(f"   writer 执行 {n_writer} 次；分析师执行 {n_analyst} 次；"
          f"人工暂停 {len(quests2)} 次")
    print(f"   human_rejects = {final2.get('human_rejects')}   "
          f"human_notes = {final2.get('human_notes')}")

    checks.append(("人工暂停两次（自动重写后 + 人工打回后）", len(quests2) == 2))
    checks.append(("writer 共执行 3 次（初稿 + 自动打回 + 人工打回）", n_writer == 3))
    checks.append(("分析师始终只跑 1 轮（未随重写重跑）", n_analyst == len(G.ROLE_SPECS)))
    checks.append(("人工意见被记入 human_notes",
                   any("数据时点" in (n or "") for n in final2.get("human_notes", []))))
    checks.append(("★ 人工意见被注入重写 prompt（独立预算生效）",
                   "人工审批人提出了以下修改要求" in inj))
    checks.append(("人工意见原文出现在 prompt 里", "补充数据时点说明" in inj))
    checks.append(("自动审查意见也注入了重写 prompt", "本轮是「重写」" in inj))
    checks.append(("最终 approve 收工",
                   (final2.get("human_decision") or "").startswith("approve")))

    # ── 场景 3：连续人工 reject → 人工熔断 ──────────────────
    print("\n【场景 3】连续人工 reject → 人工预算用尽后熔断")
    CRITIQUE_PLAN[:] = [True, True, True, True]   # 审查一路通过，排除自动熔断干扰
    critique_calls[0] = 0
    writer_prompts.clear()
    orig_h = G.MAX_HUMAN_REWRITES
    G.MAX_HUMAN_REWRITES = 1                      # 人工只允许重写 1 次
    app3 = G.build_graph()
    final3, quests3 = drive(app3, "t-human-fuse",
                            ["reject:第一次打回", "reject:第二次打回", "reject:第三次打回"])
    G.MAX_HUMAN_REWRITES = orig_h
    checks.append(("人工打回有独立预算（不占用 MAX_REVISIONS）",
                   final3.get("revision", 0) > G.MAX_REVISIONS - 1))
    checks.append((f"人工预算用尽后熔断（打回 {final3.get('human_rejects')} 次后收工）",
                   (final3.get("human_rejects") or 0) > 1 and len(quests3) == 2))
    print(f"   人工暂停 {len(quests3)} 次；human_rejects={final3.get('human_rejects')}；"
          f"revision={final3.get('revision')}（MAX_REVISIONS={G.MAX_REVISIONS}）")

    # ── 断言汇总 ────────────────────────────────────────────
    print("\n[断言]")
    all_ok = True
    for label, good in checks:
        all_ok &= bool(good)
        print(f"   {'✅' if good else '❌'} {label}")
    print("\n" + "=" * 74)
    print("  ✅ 人工确认关控制流全部正确：暂停 → 答复 → 恢复 → 打回 → 重写 → 熔断"
          if all_ok else "  ❌ 有断言失败，见上")
    print("=" * 74)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

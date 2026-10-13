#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 2：显式 StateGraph —— 一个分析师的完整流水线

这个文件存在的意义：**把 my-first-agent 里 `create_agent()` 藏起来的东西摊开。**

    create_agent(model, tools, system_prompt)        ← 一行；循环/状态/路由全藏起来
                     ↓ 它内部其实就是
    StateGraph(State) → add_node → add_edge → compile()   ← 显式、可读、能画出来

本阶段只有「一个分析师」，所以图很短：

    START → collect_facts → analyze → END

阶段 3 会把 `analyze` 拆成 4 个视角的节点 + Supervisor 调度。

═══════════════════════════════════════════════════════════
设计决策：取数是确定性的，不让 LLM 决定调哪个工具
═══════════════════════════════════════════════════════════
理由：复盘需要哪些数据是**已知的固定集合**（梯队、抱团、宽度、板块…）。
让 LLM 自己选工具会引入不确定性，而且它可能漏调关键工具，导致报告缺项。

**企业级做法：确定性取数 + LLM 只做解释。**

（对比 my-first-agent：那里是开放式问答，"用户会问什么"不可预知，
工具选择必须交给 LLM —— 场景不同，选择就不同。这个对比本身就是个知识点。）

═══════════════════════════════════════════════════════════
运行
═══════════════════════════════════════════════════════════
    python3 graph.py                  # 默认复盘日 2026-10-12
    python3 graph.py 2026-10-12

产物（写到 out/，已在 .gitignore 里排除）
==========================================
| 文件 | 内容 |
|---|---|
| `out/review-<date>.md` | 主报告 |
| `out/facts-<date>.json` | 事实包 + 口径校验 + 局限 + 用量（**可核查**） |
| `out/architecture.mmd` | 图的 mermaid 源码（架构图，阶段 7 渲染） |
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

import tools as T
from prompts import ANALYST_SYSTEM, build_user_prompt

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "out"

# 只从本项目目录读 .env —— 不跨项目读别人的密钥
load_dotenv(PROJECT_DIR / ".env")

DEFAULT_DATE = "2026-10-12"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("ANALYST_TEMPERATURE", "0.2"))


# ════════════════════════════════════════════════════════════
#  状态：显式声明
# ════════════════════════════════════════════════════════════

class ReplayState(TypedDict):
    """图的显式状态。

    对比 my-first-agent：那里是一个模块级 `conversation` 列表 + 隐式约定
    （谁往里写、写什么格式，全靠人记住）。这里每个字段都写明白。

    注意 `Annotated[..., operator.add]` —— 这叫 **reducer**：
    它决定"多个节点同时写这个字段时会怎样"。
      · 带 reducer → **追加**（limitations / trace 就该追加）
      · 不带 reducer → **覆盖**（facts / report 就该覆盖）
    阶段 3 并行扇出时，reducer 是让 4 个分析师互不覆盖的关键。
    """

    date: str
    facts: dict
    checks: dict
    limitations: Annotated[list[str], operator.add]   # 追加
    report: str                                        # 覆盖
    usage: dict
    trace: Annotated[list[dict], operator.add]         # 追加（可观测性）


# ════════════════════════════════════════════════════════════
#  节点 1：确定性取数（无 LLM，不花钱）
# ════════════════════════════════════════════════════════════

def collect_facts(state: ReplayState) -> dict:
    """调用工具层，打包成事实包。

    LLM 从头到尾**不接触原始行情数据** —— 它只拿到工具结论，
    所以没有幻觉的原材料。这是本项目最重要的安全设计。
    """
    t0 = time.time()
    date = state["date"]

    facts: dict = {}
    checks: dict = {}
    limitations: list[str] = []

    plan = [
        ("数据边界", T.data_boundaries),
        ("连板梯队", T.ladder),
        ("抱团度", T.seal_strength),
        ("竞价封板", T.auction_sealed),
        ("炸板率", T.blow_up_rate),
        ("市场宽度", T.market_breadth),
        ("行业分布", T.industry_breakdown),
    ]
    for label, fn in plan:
        r = fn(date)
        facts[label] = r["data"]
        checks[label] = r["_checks"]
        limitations.extend(r["limitations"])

    # 龙头质量：**刻意制造「强 vs 弱」的对比**，而不是只挑封单最大的几只。
    # 理由：只挑最大的，永远挑不到"烂板"，而"同样 6 板，含金量差一个量级"
    # 恰恰是复盘里最有信息量的一句话。
    # 明细已按 `封单占流通_pct` 升序排好 → 首个就是该高度最弱的一只。
    targets: list[str] = []
    ladder = facts["连板梯队"]["梯队"]          # 已按高度降序，首个键 = 最高板
    if ladder:
        heights = list(ladder)
        targets.append(ladder[heights[0]]["个股"][0])              # ① 总龙头
        if ladder[heights[0]]["家数"] > 1:
            targets.append(ladder[heights[0]]["明细"][0]["名称"])   # ② 同高度最弱（烂板）
        if len(heights) > 1:
            # 次高板取封单占流通最小的两只 —— 10/12 正好是九阳股份 + 时代万恒，
            # 与原报告「同是 6 板，含金量差一个量级」的对比表一致。
            targets += [x["名称"] for x in ladder[heights[1]]["明细"][:2]]
    targets += [x["名称"] for x in facts["抱团度"]["封单TOP榜"][:2]]  # ④ 封单最大的
    targets = list(dict.fromkeys(targets))[:5]

    facts["龙头质量"] = {}
    for nm in targets:
        r = T.leader_quality(nm, date)
        facts["龙头质量"][nm] = r["data"]
        checks[f"龙头质量:{nm}"] = r["_checks"]
        limitations.extend(r["limitations"])

    # 多个工具会重复同一条板块局限，去重保序
    limitations = list(dict.fromkeys(limitations))

    return {
        "facts": facts,
        "checks": checks,
        "limitations": limitations,
        "trace": [{
            "step": "collect_facts",
            "elapsed_s": round(time.time() - t0, 3),
            "tools_called": len(plan) + len(targets),
            "limitations_found": len(limitations),
            "llm_calls": 0,
        }],
    }


# ════════════════════════════════════════════════════════════
#  节点 2：LLM 分析师（整个流程里唯一花钱的一步）
# ════════════════════════════════════════════════════════════

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=TEMPERATURE,
    )


def analyze(state: ReplayState) -> dict:
    """分析师节点：读事实包 → 写复盘报告。"""
    t0 = time.time()
    messages = [
        SystemMessage(ANALYST_SYSTEM),
        HumanMessage(build_user_prompt(
            state["date"], state["facts"], state["checks"], state["limitations"],
        )),
    ]
    resp = _build_llm().invoke(messages)

    u = getattr(resp, "usage_metadata", None) or {}
    usage = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "total_tokens": u.get("total_tokens"),
    }

    return {
        "report": resp.content,
        "usage": usage,
        "trace": [{
            "step": "analyze",
            "elapsed_s": round(time.time() - t0, 3),
            "prompt_chars": sum(len(str(m.content)) for m in messages),
            "report_chars": len(resp.content),
            "total_tokens": usage.get("total_tokens"),
            "llm_calls": 1,
        }],
    }


# ════════════════════════════════════════════════════════════
#  建图：这几行就是 create_agent() 内部做的事
# ════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(ReplayState)
    g.add_node("collect_facts", collect_facts)
    g.add_node("analyze", analyze)
    g.add_edge(START, "collect_facts")
    g.add_edge("collect_facts", "analyze")
    g.add_edge("analyze", END)
    return g.compile()


# ════════════════════════════════════════════════════════════
#  费用估算
# ════════════════════════════════════════════════════════════

def estimate_cost(usage: dict) -> float | None:
    """按官方定价估算费用（元）。

    ⚠️ **不硬编码单价**：模型价格随时期变动，写死必然过期。
    要算费用就在 .env 里填 DEEPSEEK_PRICE_IN_PER_M / DEEPSEEK_PRICE_OUT_PER_M
    （单位：元 / 百万 token），自己从官方定价页抄。
    """
    pi, po = os.getenv("DEEPSEEK_PRICE_IN_PER_M"), os.getenv("DEEPSEEK_PRICE_OUT_PER_M")
    if not (pi and po):
        return None
    try:
        return round(
            int(usage["input_tokens"]) / 1e6 * float(pi)
            + int(usage["output_tokens"]) / 1e6 * float(po),
            4,
        )
    except (TypeError, ValueError, KeyError):
        return None


# ════════════════════════════════════════════════════════════
#  产物落盘
# ════════════════════════════════════════════════════════════

def save_artifacts(app, final: dict, date: str) -> list[Path]:
    OUT_DIR.mkdir(exist_ok=True)
    written: list[Path] = []

    p = OUT_DIR / f"review-{date}.md"
    p.write_text(final["report"], encoding="utf-8")
    written.append(p)

    p = OUT_DIR / f"facts-{date}.json"
    p.write_text(json.dumps({
        "date": date,
        "facts": final["facts"],
        "checks": final["checks"],
        "limitations": final["limitations"],
        "usage": final.get("usage"),
        "cost_yuan": estimate_cost(final.get("usage") or {}),
        "trace": final.get("trace"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(p)

    # 架构图源码 —— 一行拿到图的 mermaid 定义，阶段 7 直接渲染
    p = OUT_DIR / "architecture.mmd"
    p.write_text(app.get_graph().draw_mermaid(), encoding="utf-8")
    written.append(p)

    return written


# ════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════

def run(date: str = DEFAULT_DATE) -> dict:
    app = build_graph()
    return app.invoke({"date": date})


def main() -> int:
    if not (PROJECT_DIR / ".env").is_file():
        print("❌ 缺少 .env")
        print("   请在你自己的终端里执行：cp .env.example .env  然后填入 DEEPSEEK_API_KEY")
        return 2
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ .env 里没有 DEEPSEEK_API_KEY")
        return 2

    date = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATE
    app = build_graph()

    print("=" * 66)
    print(f"  阶段 2 · 单分析师 StateGraph   复盘日 = {date}")
    print("=" * 66)
    print("\n[图的 mermaid 定义]\n")
    print(app.get_graph().draw_mermaid())

    t0 = time.time()
    final = app.invoke({"date": date})
    elapsed = time.time() - t0

    print("\n[执行轨迹]")
    for step in final.get("trace", []):
        mark = "💰" if step.get("llm_calls") else "🧮"
        print(f"  {mark} {step['step']:14} {step['elapsed_s']:>6.3f}s  "
              f"{ {k: v for k, v in step.items() if k not in ('step', 'elapsed_s')} }")

    u = final.get("usage") or {}
    print(f"\n[用量] model={u.get('model')}  in={u.get('input_tokens')}  "
          f"out={u.get('output_tokens')}  total={u.get('total_tokens')}")
    cost = estimate_cost(u)
    print(f"[费用] {'未配置单价（在 .env 里填 DEEPSEEK_PRICE_*_PER_M 即可估算）' if cost is None else f'{cost} 元'}")

    print(f"\n[汇总] 工具调用 {sum(s.get('tools_called', 0) for s in final.get('trace', []))} 次，"
          f"LLM 调用 {sum(s.get('llm_calls', 0) for s in final.get('trace', []))} 次，"
          f"局限 {len(final.get('limitations', []))} 条，总耗时 {elapsed:.2f}s")

    files = save_artifacts(app, final, date)
    print("\n[产物]")
    for f in files:
        print(f"  {f.relative_to(PROJECT_DIR)}  ({f.stat().st_size} 字节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

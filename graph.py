#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 3（提前做）：反方审查员 + reflection 回路

图的结构从"一条直线"变成"带循环的分支"：

    START → collect_facts → analyze → critique ─┬─(有问题 且 revision < MAX) → analyze
                                                 └─(通过 或 达上限) ─────────→ END

这 4 行就是 `create_agent()` 内部那个"循环"的本质 ——
**条件边 + 回到前一个节点**。做完这个，Agent 循环对你就不再是黑盒了。

═══════════════════════════════════════════════════════════
为什么需要审查员（来自真实实测）
═══════════════════════════════════════════════════════════
实测发现三类错误 **数字全对、错在表述层**：

| 错误 | 报告原文 |
|---|---|
| 自相矛盾 | 「8 家里有 **5 家**是连板股」，括号里却列出 6 只 |
| 比较方向反 | 「东百 3.21 亿**仅次于**赛力斯 3.1 亿」（其实东百更大） |
| 计数错 | 「这 **4** 个高度」（7/8/9 只有 3 个） |

这三类靠"叮嘱分析师仔细点"能压一时，但不可靠。
需要一个**独立视角**拿着成品挑毛病 —— 而且它只拿到报告 + 事实包，
**仍不接触原始行情数据**，所以它也不能凭空编造。

═══════════════════════════════════════════════════════════
设计决策：取数是确定性的，不让 LLM 决定调哪个工具
═══════════════════════════════════════════════════════════
复盘需要哪些数据是**已知的固定集合**。让 LLM 自己选工具会引入不确定性，
还可能漏调关键工具导致报告缺项。→ **确定性取数 + LLM 只做解释。**

（对比 my-first-agent：那里是开放式问答，"用户会问什么"不可预知，
工具选择必须交给 LLM —— 场景不同，选择就不同。）

═══════════════════════════════════════════════════════════
运行
═══════════════════════════════════════════════════════════
    python3 graph.py                  # 默认复盘日 2026-10-12
    MAX_REVISIONS=1 python3 graph.py  # 控制最多重写几次（默认 2）

产物（写到 out/，已在 .gitignore 里排除）
==========================================
| 文件 | 内容 |
|---|---|
| `out/review-<date>.md` | 主报告（含审查通过后的终稿） |
| `out/facts-<date>.json` | 事实包 + 口径校验 + 局限 + 审查记录 + 用量 |
| `out/architecture.mmd` | 图的 mermaid 源码（架构图） |
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
from pydantic import BaseModel, Field

import tools as T
from prompts import (
    ANALYST_SYSTEM,
    CRITIC_SYSTEM,
    build_critic_prompt,
    build_user_prompt,
)

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "out"

# 只从本项目目录读 .env —— 不跨项目读别人的密钥
load_dotenv(PROJECT_DIR / ".env")

DEFAULT_DATE = "2026-10-12"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("ANALYST_TEMPERATURE", "0.2"))
# 最多生成几版报告（1 = 不重写；2 = 允许打回一次）。这是最早的「步数熔断」雏形：
# 没有它，审查员和分析师可能互相不服，无限循环烧钱。
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))


# ════════════════════════════════════════════════════════════
#  状态：显式声明
# ════════════════════════════════════════════════════════════

class ReplayState(TypedDict):
    """图的显式状态。

    对比 my-first-agent：那里是一个模块级 `conversation` 列表 + 隐式约定。
    这里每个字段都写明白。

    `Annotated[..., operator.add]` 叫 **reducer**，决定"多个节点写同一字段时怎样"：
      · 带 reducer → **追加**（limitations / trace / usages / critique）
      · 不带 reducer → **覆盖**（facts / report / revision）
    本图有回环，`analyze` 会被执行多次 —— 若 `trace`、`usages` 不带 reducer，
    第二次执行的记录就会把第一次的覆盖掉。
    """

    date: str
    facts: dict
    checks: dict
    limitations: Annotated[list[str], operator.add]   # 追加
    report: str                                        # 覆盖（每次重写替换）
    critique: Annotated[list[dict], operator.add]      # 追加：每轮审查各一条
    revision: int                                      # 覆盖：当前报告版本号（1 = 初稿）
    usages: Annotated[list[dict], operator.add]        # 追加：每次 LLM 调用一条
    trace: Annotated[list[dict], operator.add]         # 追加：可观测性


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
#  LLM 工厂 + 用量记录
# ════════════════════════════════════════════════════════════

def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=TEMPERATURE,
    )


def _usage_of(message, role: str, version: int) -> dict:
    u = getattr(message, "usage_metadata", None) or {}
    return {
        "role": role,
        "version": version,
        "model": MODEL,
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "total_tokens": u.get("total_tokens"),
    }


# ════════════════════════════════════════════════════════════
#  节点 2：LLM 分析师（会被执行 1~MAX_REVISIONS 次）
# ════════════════════════════════════════════════════════════

def analyze(state: ReplayState) -> dict:
    """分析师节点：读事实包 → 写复盘报告。

    若是重写（`critique` 非空），会把上一轮审查意见一并送进去，
    要求**逐条修掉**而不是整篇推倒重写。
    """
    t0 = time.time()
    prior = state.get("critique") or []
    version = state.get("revision", 0) + 1

    messages = [
        SystemMessage(ANALYST_SYSTEM),
        HumanMessage(build_user_prompt(
            state["date"], state["facts"], state["checks"], state["limitations"],
            critique=prior or None,
        )),
    ]
    resp = _build_llm().invoke(messages)
    usage = _usage_of(resp, "analyst", version)

    return {
        "report": resp.content,
        "revision": version,
        "usages": [usage],
        "trace": [{
            "step": f"analyze#{version}{'（重写）' if prior else ''}",
            "elapsed_s": round(time.time() - t0, 3),
            "is_rewrite": bool(prior),
            "prompt_chars": sum(len(str(m.content)) for m in messages),
            "report_chars": len(resp.content),
            "total_tokens": usage["total_tokens"],
            "llm_calls": 1,
        }],
    }


# ════════════════════════════════════════════════════════════
#  节点 3：反方审查员（结构化输出）
# ════════════════════════════════════════════════════════════

class _Finding(BaseModel):
    """一条审查发现。"""
    kind: str = Field(description="问题类型：数字不一致 / 比较方向 / 计数错误 / 漏免责声明 / 买卖措辞 / 无依据数字 / 断言超口径 / 其他")
    location: str = Field(description="报告中的位置：小节名 + 引用原句片段")
    detail: str = Field(description="问题具体是什么")
    fix_hint: str = Field(description="可执行的修改方向")


class _Critique(BaseModel):
    """审查结论（用 Pydantic 约束结构，而不是靠解析自由文本）。"""
    ok: bool = Field(description="报告是否可直接交付（无阻断性问题）")
    summary: str = Field(description="一句话总评")
    findings: list[_Finding] = Field(default_factory=list,
                                     description="发现的问题列表；没有问题就留空数组")


def critique(state: ReplayState) -> dict:
    """独立视角挑毛病。

    `include_raw=True` 让我们**同时拿到结构化结果和原始 AIMessage** ——
    后者才有 usage_metadata，否则拿不到这一次的 token 用量。
    """
    t0 = time.time()
    version = state.get("revision", 1)
    messages = [
        SystemMessage(CRITIC_SYSTEM),
        HumanMessage(build_critic_prompt(
            state["date"], state["report"], state["facts"],
            state["checks"], state["limitations"],
        )),
    ]

    verdict: dict
    usage: dict = {}
    out, last_err = None, None

    # ⚠️ 实测踩坑（跨厂商兼容性）：langchain-openai 的 `with_structured_output`
    #    **默认 method 是 json_schema**，而 DeepSeek 不支持它，会报 400：
    #        "This response_format type is unavailable now"
    #    而 `function_calling` 方式是可用的。所以这里**显式指定**并按顺序兜底，
    #    全部失败才降级 —— 绝不让一个"审查能力不可用"变成流程中断。
    for method in ("function_calling", "json_mode"):
        try:
            out = _build_llm().with_structured_output(
                _Critique, method=method, include_raw=True
            ).invoke(messages)
            break
        except Exception as e:
            last_err = e
            out = None

    if out is None:                              # 所有方式都失败 → 降级但不断链
        verdict = {
            "ok": True,
            "summary": f"⚠️ 审查降级：结构化输出失败"
                       f"（{type(last_err).__name__}: {str(last_err)[:90]}）",
            "findings": [], "degraded": True,
        }
    else:
        msg = out.get("raw")
        if msg is not None:
            usage = _usage_of(msg, "critic", version)
        parsed = out.get("parsed")
        if parsed is None:                       # 拿到了回复但解析失败 → 同样降级
            verdict = {"ok": True,
                       "summary": f"⚠️ 审查降级：结构化解析失败（{out.get('parsing_error')}）",
                       "findings": [], "degraded": True}
        else:
            verdict = {"ok": parsed.ok, "summary": parsed.summary,
                       "findings": [f.model_dump() for f in parsed.findings]}

    verdict["reviewed_version"] = version
    return {
        "critique": [verdict],
        "usages": [usage] if usage else [],
        "trace": [{
            "step": f"critique#{version}",
            "elapsed_s": round(time.time() - t0, 3),
            "ok": verdict["ok"],
            "findings": len(verdict["findings"]),
            "degraded": verdict.get("degraded", False),
            "total_tokens": usage.get("total_tokens"),
            "llm_calls": 1,
        }],
    }


def route_after_critique(state: ReplayState) -> str:
    """**条件边**：审查不过就打回重写，但受 MAX_REVISIONS 熔断。

    这就是 `create_agent()` 里"循环"的本质 —— 一个函数根据状态决定下一步去哪。
    """
    if not state.get("critique"):
        return "end"
    if state["critique"][-1].get("ok"):
        return "end"
    if state.get("revision", 1) >= MAX_REVISIONS:
        return "end"          # 熔断：不能无限重写
    return "rewrite"


# ════════════════════════════════════════════════════════════
#  建图：这几行就是 create_agent() 内部做的事
# ════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(ReplayState)
    g.add_node("collect_facts", collect_facts)
    g.add_node("analyze", analyze)
    g.add_node("critique", critique)

    g.add_edge(START, "collect_facts")
    g.add_edge("collect_facts", "analyze")
    g.add_edge("analyze", "critique")
    # 条件边：critique 之后要么回到 analyze（重写），要么收工
    g.add_conditional_edges("critique", route_after_critique,
                            {"rewrite": "analyze", "end": END})
    return g.compile()


# ════════════════════════════════════════════════════════════
#  汇总与费用
# ════════════════════════════════════════════════════════════

def total_usage(usages: list[dict]) -> dict:
    """把多次调用的用量加总（本图最多 2×analyze + 2×critique = 4 次）。"""
    return {
        "calls": len(usages),
        "model": usages[0].get("model") if usages else None,
        "input_tokens": sum(u.get("input_tokens") or 0 for u in usages),
        "output_tokens": sum(u.get("output_tokens") or 0 for u in usages),
        "total_tokens": sum(u.get("total_tokens") or 0 for u in usages),
        "breakdown": [{"role": u.get("role"), "version": u.get("version"),
                       "total_tokens": u.get("total_tokens")} for u in usages],
    }


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
    """落盘产物。允许"部分状态"—— LLM 失败时也要保住取数成果。"""
    OUT_DIR.mkdir(exist_ok=True)
    written: list[Path] = []

    # 报告可能不存在（analyze 失败），此时跳过而不是写一个空文件
    if final.get("report"):
        p = OUT_DIR / f"review-{date}.md"
        p.write_text(final["report"], encoding="utf-8")
        written.append(p)

    p = OUT_DIR / f"facts-{date}.json"
    p.write_text(json.dumps({
        "date": date,
        "facts": final.get("facts", {}),
        "checks": final.get("checks", {}),
        "limitations": final.get("limitations", []),
        "revision": final.get("revision"),
        "critique": final.get("critique", []),
        "usage": total_usage(final.get("usages", [])),
        "cost_yuan": estimate_cost(total_usage(final.get("usages", []))),
        "trace": final.get("trace"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(p)

    # 架构图源码 —— 一行拿到图的 mermaid 定义
    p = OUT_DIR / "architecture.mmd"
    p.write_text(app.get_graph().draw_mermaid(), encoding="utf-8")
    written.append(p)

    return written


# ════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════

def _diagnose(err: Exception) -> None:
    """把异常翻译成人能看懂的一句话，而不是甩 50 行 traceback。"""
    msg = str(err)
    print(f"\n❌ 分析失败：{type(err).__name__}")
    low = msg.lower()
    if "401" in msg or "authentication" in low:
        print("   原因：API Key 无效，或还是 .env.example 里的占位符。")
        print("   处理：nano .env  → 把 DEEPSEEK_API_KEY= 右边换成你的真实 key")
    elif any(k in msg for k in ("Connection", "Timeout", "timed out", "getaddrinfo")):
        print("   原因：连不上端点。检查网络，或核对 .env 里的 DEEPSEEK_BASE_URL。")
    elif "429" in msg or "rate limit" in low:
        print("   原因：触发限流。稍等再试。")
    else:
        print(f"   {msg[:400]}")


def _print_review_chain(final: dict) -> None:
    """把审查回路的过程单独打出来 —— 这是本项目最有观赏性的一段。"""
    critique = final.get("critique", [])
    if not critique:
        return
    print(f"\n[审查回路] 共 {len(critique)} 轮，最终报告版本 v{final.get('revision')}")
    for i, c in enumerate(critique, 1):
        mark = "✅" if c.get("ok") else "❌"
        deg = "（降级）" if c.get("degraded") else ""
        print(f"  {mark} 第{i}轮 审 v{c.get('reviewed_version')}{deg}：{c.get('summary', '')[:56]}")
        for f in c.get("findings", []):
            print(f"       · [{f.get('kind')}] {f.get('detail', '')[:74]}")


def _print_run_summary(final: dict, elapsed: float) -> None:
    print("\n[执行轨迹]")
    for step in final.get("trace", []):
        mark = "💰" if step.get("llm_calls") else "🧮"
        extra = {k: v for k, v in step.items() if k not in ("step", "elapsed_s")}
        print(f"  {mark} {step['step']:16} {step['elapsed_s']:>6.3f}s  {extra}")

    _print_review_chain(final)

    u = total_usage(final.get("usages", []))
    print(f"\n[用量] {u['calls']} 次 LLM 调用  model={u['model']}  "
          f"in={u['input_tokens']}  out={u['output_tokens']}  total={u['total_tokens']}")
    for b in u["breakdown"]:
        print(f"        · {b['role']}#{b['version']}  {b['total_tokens']} tokens")
    cost = estimate_cost(u)
    print("[费用] " + ("未配置单价（在 .env 里填 DEEPSEEK_PRICE_*_PER_M 即可估算）"
                      if cost is None else f"{cost} 元"))
    print(f"\n[汇总] 工具调用 {sum(s.get('tools_called', 0) for s in final.get('trace', []))} 次，"
          f"LLM 调用 {u['calls']} 次，"
          f"局限 {len(final.get('limitations', []))} 条，总耗时 {elapsed:.2f}s")


def main() -> int:
    if not (PROJECT_DIR / ".env").is_file():
        print("❌ 缺少 .env")
        print("   请在你自己的终端里执行：cp .env.example .env")
        print("   然后编辑 .env，把 DEEPSEEK_API_KEY= 右边换成你的真实 key")
        return 2
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ .env 里没有 DEEPSEEK_API_KEY")
        return 2

    date = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATE
    app = build_graph()

    print("=" * 70)
    print(f"  反方审查员 + reflection 回路   复盘日 = {date}   最多重写 {MAX_REVISIONS - 1} 次")
    print("=" * 70)
    print("\n[图的 mermaid 定义]\n")
    print(app.get_graph().draw_mermaid())

    t0 = time.time()
    final: dict = {}
    try:
        # 用 stream_mode="values"：每个节点跑完都给一份**完整状态快照**。
        # 好处很实际 —— 后面节点失败时，前面节点（取数）的成果不会丢。
        # 对比 app.invoke()：一抛异常，前面所有成果一起没了（我们实测踩过）。
        for chunk in app.stream({"date": date}, stream_mode="values"):
            final = chunk
    except Exception as e:
        _diagnose(e)
        if final.get("trace"):
            print("\n   ↳ 已完成节点的成果已保住，不重跑取数")
            _print_run_summary(final, time.time() - t0)
        print("\n[产物]（部分）")
        for f in save_artifacts(app, final, date):
            print(f"  {f.relative_to(PROJECT_DIR)}  ({f.stat().st_size} 字节)")
        return 1

    elapsed = time.time() - t0
    _print_run_summary(final, elapsed)

    print("\n[产物]")
    for f in save_artifacts(app, final, date):
        print(f"  {f.relative_to(PROJECT_DIR)}  ({f.stat().st_size} 字节)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

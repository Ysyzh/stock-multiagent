#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 3：4 个视角分析师并行扇出 + Supervisor 汇总 + Writer 执笔 + 审查回路

完整图（★ 是阶段 3 新增的部分）：

                       ┌──────────────┐
        START ──► collect_facts       │  确定性取数（无 LLM）
                       └──────┬───────┘
             ┌────────────┬───┴────────┬────────────┐   ★ 并行扇出
             ▼            ▼            ▼            ▼
      ★ sentiment   ★ leader    ★ handover    ★ sector      4 个视角，各拿一片事实包
             └────────────┴───┬────────┴────────────┘
                              ▼
                       ★ supervisor          汇总：找冲突 / 排优先级 / 指缺口
                              ▼
                         analyze              执笔：产出九节报告
                              ▼
                         critique ─┬─(不通过且 revision<2)→ analyze
                                   └─(通过或达上限)───────→ END

═══════════════════════════════════════════════════════════
阶段 3 的两个关键设计
═══════════════════════════════════════════════════════════

**1. 事实包切片 —— 每个分析师只看到它需要的那几块。**

   全量事实包 JSON 约 3.3k 字符。若 4 个分析师人手一份，input 直接 ×4。
   切片后每个约 1k 字符，代价从"4 倍"降到"1.3 倍"左右。

   > 工程上这句话叫「让每个 Agent 只看到它需要的信息」。
   > 它同时降低成本和**降低噪音** —— 视角分析师看到无关板块反而会分心。

**2. reducer 在这里才真正发挥作用。**

   `findings` 带 `Annotated[..., operator.add]`。4 个分析师**并行**写同一个字段，
   reducer 保证它们**追加**而不是互相覆盖。
   阶段 2 里 reducer 只是个演示；到了并行扇出，**没有它就会丢数据**。

**3. 重写只回到 `analyze`，不重跑分析师。**

   审查意见针对的是**成稿**，4 份视角结论仍然有效。重跑分析师既慢又贵且无必要。
   （代价写在 README 的取舍说明里：若审查发现某条视角结论本身错了，
     当前实现不会自动重跑该视角。）

运行
====
    python3 graph.py                  # 默认复盘日 2026-10-12
    MAX_REVISIONS=1 python3 graph.py  # 收紧熔断
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

import tools as T
from prompts import (
    CRITIC_SYSTEM,
    SUPERVISOR_SYSTEM,
    WRITER_SYSTEM,
    build_critic_prompt,
    build_role_prompt,
    build_role_system,
    build_supervisor_prompt,
    build_writer_prompt,
)

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "out"

# 只从本项目目录读 .env —— 不跨项目读别人的密钥
load_dotenv(PROJECT_DIR / ".env")

DEFAULT_DATE = "2026-10-12"
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = float(os.getenv("ANALYST_TEMPERATURE", "0.2"))
# 最多生成几版报告（1 = 不重写）。这是最早的「步数熔断」雏形：
# 没有它，审查员和执笔人可能互相不服，无限循环烧钱。
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
# 人工确认关：默认开启。非交互环境（脚本/CI）会自动通过并打印告警。
# 设为 0 可整体关闭这一关。
HUMAN_REVIEW = os.getenv("HUMAN_REVIEW", "1") != "0"
# 人工打回的**独立预算**。刻意与 MAX_REVISIONS 分开 ——
# 实测踩到过：自动审查回路刚好用满额度时，route 用 revision 判熔断，
# 导致**人的打回意见被无声忽略**。人和机器应该是两套预算。
MAX_HUMAN_REWRITES = int(os.getenv("MAX_HUMAN_REWRITES", "2"))


# ════════════════════════════════════════════════════════════
#  4 个视角的配置 —— 每个视角只拿自己需要的「事实包板块」
# ════════════════════════════════════════════════════════════
#  slices 里的名字必须与 collect_facts() 里的板块名一致。
#  切片是**按板块**切的，不是按字段 —— 这样每个数字仍带着它的 `_checks`，
#  分析师看到的口径信息不会比全量版少。

ROLE_SPECS: dict[str, dict] = {
    "sentiment": {
        "label": "情绪分析师",
        "focus": "市场情绪温度与宽度：连板高度与梯队断档、炸板率、涨跌分布",
        "slices": ["数据边界", "连板梯队", "炸板率", "市场宽度", "竞价封板"],
        # 只需要梯队的高度/家数/个股，不需要每只票的封单与换手明细
        "drop_ladder_detail": True,
    },
    "leader": {
        "label": "龙头质量分析师",
        "focus": "封单资金结构与龙头封板质量三指标（封单占流通市值 / 换手率 / 炸板次数）",
        "slices": ["抱团度", "龙头质量"],
    },
    "handover": {
        "label": "换龙接力分析师",
        "focus": "梯队结构的承接力：中间层是否空心、2 板→3 板的晋级通道是否通畅",
        # 判断"承接力"必须看每只票的封单占流通，**保留**明细
        "slices": ["连板梯队", "炸板率"],
    },
    "sector": {
        "label": "板块带动分析师",
        "focus": "涨停股的板块分布与集中度（务必说明行业口径 ≠ 题材口径）",
        "slices": ["行业分布", "连板梯队"],
        "drop_ladder_detail": True,
    },
}


def _trim_slice(slice_facts: dict, spec: dict) -> dict:
    """按视角需要做**字段级**裁剪。

    「事实包切片」分两层：
      ① **板块级**：只给这个视角相关的板块（`slices`）
      ② **字段级**：板块里过细的字段也可以砍掉（`drop_ladder_detail`）

    连板梯队的 `明细` 里，每个高度每只票都带封单/换手/炸板 —— 对"情绪温度"
    和"板块分布"这两个视角是纯噪音。砍掉它既省 token 又降噪。
    """
    if not spec.get("drop_ladder_detail"):
        return slice_facts
    import copy
    out = copy.deepcopy(slice_facts)
    lad = out.get("连板梯队")
    if isinstance(lad, dict) and isinstance(lad.get("梯队"), dict):
        for item in lad["梯队"].values():
            if isinstance(item, dict):
                item.pop("明细", None)
    return out


# ════════════════════════════════════════════════════════════
#  状态：显式声明
# ════════════════════════════════════════════════════════════

class ReplayState(TypedDict):
    """图的显式状态。

    `Annotated[..., operator.add]` 叫 **reducer**，决定"多个节点写同一字段时怎样"：
      · 带 reducer → **追加**（limitations / trace / usages / critique / **findings**）
      · 不带 reducer → **覆盖**（facts / digest / report / revision）

    ⚠️ `findings` 是本图里 reducer **不可省**的字段：4 个分析师在**同一个超步**
    里并行写它，没有 reducer 就会互相覆盖，最后只剩一份视角结论。
    """

    date: str
    facts: dict
    checks: dict
    limitations: Annotated[list[str], operator.add]     # 追加
    findings: Annotated[list[dict], operator.add]       # ★ 追加：4 个视角各写一条
    digest: str                                          # 覆盖：主管汇总
    report: str                                          # 覆盖：每次重写替换
    critique: Annotated[list[dict], operator.add]        # 追加：每轮审查各一条
    revision: int                                        # 覆盖：当前报告版本号
    human_notes: Annotated[list[str], operator.add]      # ★ 追加：人工审批给出的修改意见
    human_decision: str                                  # 覆盖：最近一次人工决定
    human_rejects: int                                   # 覆盖：人工打回次数（独立于 revision）
    usages: Annotated[list[dict], operator.add]          # 追加：每次 LLM 调用一条
    trace: Annotated[list[dict], operator.add]           # 追加：可观测性


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
    # 顺手记下 prompt 缓存命中量。实测 DeepSeek 会自动缓存公共前缀
    # （writer 与 critic 共享大段 system prompt），命中部分通常不计费或大幅折扣。
    details = u.get("input_token_details") or {}
    return {
        "role": role,
        "version": version,
        "model": MODEL,
        "input_tokens": u.get("input_tokens"),
        "output_tokens": u.get("output_tokens"),
        "total_tokens": u.get("total_tokens"),
        "cache_read": details.get("cache_read"),
    }


# ════════════════════════════════════════════════════════════
#  节点 2：4 个视角分析师（★ 并行扇出）
# ════════════════════════════════════════════════════════════

def _make_analyst_node(key: str, spec: dict):
    """工厂：按视角配置生成一个分析节点函数。

    每个节点都**只把自己那片切片**交给 LLM —— 这就是"事实包切片"的落地方式。
    """
    def node(state: ReplayState) -> dict:
        t0 = time.time()
        all_facts, all_checks = state["facts"], state["checks"]
        slice_facts = _trim_slice(
            {k: all_facts[k] for k in spec["slices"] if k in all_facts}, spec
        )
        slice_checks = {k: all_checks[k] for k in spec["slices"] if k in all_checks}

        messages = [
            SystemMessage(build_role_system(spec["label"], spec["focus"])),
            HumanMessage(build_role_prompt(
                state["date"], spec["label"], spec["focus"], slice_facts, slice_checks,
            )),
        ]
        resp = _build_llm().invoke(messages)
        usage = _usage_of(resp, f"analyst:{key}", 1)

        slice_chars = len(json.dumps(slice_facts, ensure_ascii=False))
        return {
            "findings": [{
                "analyst": key,
                "label": spec["label"],
                "focus": spec["focus"],
                "slices_used": list(slice_facts),
                "text": resp.content,
            }],
            "usages": [usage],
            "trace": [{
                "step": f"analyst:{key}",
                "elapsed_s": round(time.time() - t0, 3),
                "slices": len(slice_facts),
                "slice_chars": slice_chars,
                "prompt_chars": sum(len(str(m.content)) for m in messages),
                "out_chars": len(resp.content),
                "total_tokens": usage["total_tokens"],
                "llm_calls": 1,
            }],
        }

    node.__name__ = f"analyst_{key}"
    return node


# ════════════════════════════════════════════════════════════
#  节点 3：Supervisor（★ 汇总扇入）
# ════════════════════════════════════════════════════════════

def supervisor(state: ReplayState) -> dict:
    """汇总 4 份视角结论：找冲突 / 排优先级 / 指缺口。

    刻意让它的输出**很短**（≤300 字）—— 它是"编排"节点，不是"写作"节点。
    冲突点是后面报告里最有价值的部分：一个说"情绪退潮"、另一个说"极端抱团"，
    这种张力不该被抹平。
    """
    t0 = time.time()
    findings = state.get("findings") or []
    messages = [
        SystemMessage(SUPERVISOR_SYSTEM),
        HumanMessage(build_supervisor_prompt(state["date"], findings)),
    ]
    resp = _build_llm().invoke(messages)
    usage = _usage_of(resp, "supervisor", 1)

    return {
        "digest": resp.content,
        "usages": [usage],
        "trace": [{
            "step": "supervisor",
            "elapsed_s": round(time.time() - t0, 3),
            "views_merged": len(findings),
            "prompt_chars": sum(len(str(m.content)) for m in messages),
            "out_chars": len(resp.content),
            "total_tokens": usage["total_tokens"],
            "llm_calls": 1,
        }],
    }


# ════════════════════════════════════════════════════════════
#  节点 4：Writer 执笔（会被执行 1~MAX_REVISIONS 次）
# ════════════════════════════════════════════════════════════

def analyze(state: ReplayState) -> dict:
    """执笔节点：完整事实包 + 4 份视角结论 + 主管汇总 → 九节报告。

    若是重写（`critique` 非空），会把上一轮审查意见一并送进去，
    要求**逐条修掉**而不是整篇推倒重写。
    """
    t0 = time.time()
    prior = state.get("critique") or []
    version = state.get("revision", 0) + 1

    messages = [
        SystemMessage(WRITER_SYSTEM),
        HumanMessage(build_writer_prompt(
            state["date"], state["facts"], state["checks"], state["limitations"],
            findings=state.get("findings") or None,
            digest=state.get("digest", ""),
            critique=prior or None,
            human_notes=state.get("human_notes") or None,
        )),
    ]
    resp = _build_llm().invoke(messages)
    usage = _usage_of(resp, "writer", version)

    return {
        "report": resp.content,
        "revision": version,
        "usages": [usage],
        "trace": [{
            "step": f"writer#{version}{'（重写）' if prior else ''}",
            "elapsed_s": round(time.time() - t0, 3),
            "is_rewrite": bool(prior),
            "prompt_chars": sum(len(str(m.content)) for m in messages),
            "report_chars": len(resp.content),
            "total_tokens": usage["total_tokens"],
            "llm_calls": 1,
        }],
    }


# ════════════════════════════════════════════════════════════
#  节点 5：反方审查员（结构化输出）
# ════════════════════════════════════════════════════════════

class _Finding(BaseModel):
    """一条审查发现。"""
    kind: str = Field(description="问题类型：数字不一致 / 比较方向 / 计数错误 / 漏免责声明 / 买卖措辞 / 无依据数字 / 断言超口径 / 其他")
    location: str = Field(description="报告中的位置：小节名 + 引用原句片段")
    detail: str = Field(description="问题具体是什么，计数类必须逐个列出你数到的东西")
    fix_hint: str = Field(description="可执行的修改方向")


class _Critique(BaseModel):
    """审查结论（用 Pydantic 约束结构，而不是靠解析自由文本）。"""
    ok: bool = Field(description="报告是否可直接交付（无阻断性问题）")
    summary: str = Field(description="一句话总评")
    findings: list[_Finding] = Field(default_factory=list,
                                     description="发现的问题列表；没有问题就留空数组")


def _call_structured(messages) -> tuple[dict | None, Exception | None]:
    """调用结构化输出，返回 `(out, last_err)`。做了两层兜底：

    **① method 依次尝试**：`function_calling` → `json_mode`。
    原因：langchain-openai 的 `with_structured_output` **默认 method 是
    `json_schema`**，而 DeepSeek 不支持它，会报 400
    "This response_format type is unavailable now"。

    **② 同一 method 内重试一次**：实测偶发 `parsed=None`
    （模型这一轮没返回 tool call），重试即可恢复。
    """
    last_err: Exception | None = None
    for method in ("function_calling", "json_mode"):
        for _ in range(2):                       # 每次 method 试 2 遍
            try:
                out = _build_llm().with_structured_output(
                    _Critique, method=method, include_raw=True
                ).invoke(messages)
            except Exception as e:
                last_err = e
                break                            # 调用失败 → 换下一种 method
            if out.get("parsed") is not None:
                return out, None                 # 解析成功
            last_err = None                      # 调用成功但解析失败 → 再试一遍
    return None, last_err


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

    out, last_err = _call_structured(messages)

    if out is None:                              # 所有方式都失败 → 降级但不断链
        why = (f"{type(last_err).__name__}: {str(last_err)[:80]}" if last_err
               else "调用成功但多次解析失败（模型未返回 tool call）")
        verdict = {"ok": True, "summary": f"⚠️ 审查降级：{why}",
                   "findings": [], "degraded": True}
        usage: dict = {}
    else:
        msg = out.get("raw")
        usage = _usage_of(msg, "critic", version) if msg is not None else {}
        parsed = out.get("parsed")
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

    重写**只回到 analyze（执笔）**，不重跑 4 个分析师 —— 审查意见针对的是成稿。
    审查通过后进入**人工确认关**（`human_review`），这是企业级的关键动作：
    「生成」和「发布」必须分开。
    """
    if not state.get("critique"):
        return "human"
    if state["critique"][-1].get("ok"):
        return "human"
    if state.get("revision", 1) >= MAX_REVISIONS:
        return "human"        # 熔断：不能无限重写
    return "rewrite"


# ════════════════════════════════════════════════════════════
#  节点 6：人工确认关（★ 阶段 5：interrupt 暂停/恢复）
# ════════════════════════════════════════════════════════════
#
#  `interrupt()` 会**暂停整个图**并把 payload 交回调用方；调用方用
#  `Command(resume=<答复>)` 恢复。这需要 **checkpointer** —— 没有它，
#  状态无处保存，也就无法恢复。
#
#  为什么要有这个节点：**「生成报告」和「发布报告」是两件事。**
#  自动生成再自动交付，在企业里是不可接受的（这份产物会被人当决策依据）。
#  所以加一道人审：人可以 approve，也可以 reject 并写下具体修改意见，
#  后者会让图**回到执笔人**按人的意见重写。

def human_review(state: ReplayState) -> dict:
    """出稿前的最后一道关。审批人可以看到：报告全文 + 最后一轮审查结论。"""
    t0 = time.time()
    last_critique = (state.get("critique") or [{}])[-1]
    payload = {
        "revision": state.get("revision"),
        "reviewed_ok": last_critique.get("ok"),
        "reviewed_degraded": last_critique.get("degraded", False),
        "critic_summary": last_critique.get("summary", ""),
        "report": state.get("report", ""),
        "question": "报告可以交付吗？回复 approve 通过，或 reject:<修改意见> 打回重写",
    }
    # ★ 图在此暂停，直到调用方 Command(resume=...) 送回答复
    answer = interrupt(payload)

    decision = str(answer or "approve").strip()
    approved = decision.lower().startswith(("approve", "y", "yes", "通过", "同意", "ok"))
    note = "" if approved else decision.split(":", 1)[-1].strip()
    rejects = state.get("human_rejects") or 0

    return {
        "human_decision": decision,
        "human_notes": [note] if note else [],
        "human_rejects": rejects if approved else rejects + 1,
        "trace": [{
            "step": f"human_review@v{state.get('revision')}",
            "elapsed_s": round(time.time() - t0, 3),
            "decision": "approve" if approved else "reject",
            "reject_no": 0 if approved else rejects + 1,
            "note_chars": len(note),
            "llm_calls": 0,
        }],
    }


def route_after_human(state: ReplayState) -> str:
    """人工审批后的分流：通过就收工，打回就回执笔人按人的意见重写。

    人工打回用**独立预算** `MAX_HUMAN_REWRITES`，不受自动审查回路的
    `MAX_REVISIONS` 影响 —— 否则"自动审查刚好用满额度"会让人的意见被无声忽略。
    """
    decision = (state.get("human_decision") or "approve").lower()
    if decision.startswith(("approve", "y", "yes", "通过", "同意", "ok")):
        return "end"
    if (state.get("human_rejects") or 0) > MAX_HUMAN_REWRITES:
        return "end"          # 人工熔断：人也不能无限打回
    return "rewrite"


# ════════════════════════════════════════════════════════════
#  建图
# ════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(ReplayState)
    g.add_node("collect_facts", collect_facts)
    g.add_edge(START, "collect_facts")        # 入口边：缺了它 compile() 会直接报错

    # ★ 4 个视角分析师：collect_facts 完成后它们处于**同一个超步**，会并行执行
    analyst_names = []
    for key, spec in ROLE_SPECS.items():
        name = f"analyst_{key}"
        g.add_node(name, _make_analyst_node(key, spec))
        g.add_edge("collect_facts", name)
        analyst_names.append(name)

    g.add_node("supervisor", supervisor)
    # ★ 扇入：supervisor 等 4 个分析师全部完成才会被执行
    for name in analyst_names:
        g.add_edge(name, "supervisor")

    g.add_node("analyze", analyze)
    g.add_node("critique", critique)
    g.add_node("human_review", human_review)

    g.add_edge("supervisor", "analyze")
    g.add_edge("analyze", "critique")
    # 条件边：审查不过回到 analyze（重写），通过则进入人工确认关
    g.add_conditional_edges("critique", route_after_critique,
                            {"rewrite": "analyze", "human": "human_review"})
    # 人工确认关：通过收工，打回则回到 analyze 按人的意见重写
    g.add_conditional_edges("human_review", route_after_human,
                            {"rewrite": "analyze", "end": END})

    # ★ 阶段 5：`interrupt()` 必须配 checkpointer —— 没有它状态无处保存，
    #   图就无法"暂停→稍后恢复"。这里用内存版；生产应换 SqliteSaver/PostgresSaver。
    return g.compile(checkpointer=MemorySaver())


# ════════════════════════════════════════════════════════════
#  汇总与费用
# ════════════════════════════════════════════════════════════

def total_usage(usages: list[dict]) -> dict:
    """把多次调用的用量加总，并**按角色**分组 + 统计 prompt 缓存命中。"""
    by_role: dict[str, dict] = {}
    for u in usages:
        r = u.get("role", "?")
        d = by_role.setdefault(r, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                   "total_tokens": 0, "cache_read": 0})
        d["calls"] += 1
        for k in ("input_tokens", "output_tokens", "total_tokens", "cache_read"):
            d[k] += u.get(k) or 0
    inp = sum(u.get("input_tokens") or 0 for u in usages)
    cached = sum(u.get("cache_read") or 0 for u in usages)
    return {
        "calls": len(usages),
        "model": usages[0].get("model") if usages else None,
        "input_tokens": inp,
        "output_tokens": sum(u.get("output_tokens") or 0 for u in usages),
        "total_tokens": sum(u.get("total_tokens") or 0 for u in usages),
        "cache_read": cached,
        # 命中率 = 缓存读到的 input / 全部 input（DeepSeek 对命中部分通常大幅折扣）
        "cache_hit_rate": round(cached / inp * 100, 1) if inp else 0.0,
        "by_role": by_role,
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

    # 报告可能不存在（执笔失败），此时跳过而不是写一个空文件
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
        "findings": final.get("findings", []),
        "digest": final.get("digest", ""),
        "revision": final.get("revision"),
        "critique": final.get("critique", []),
        # 人工审批的审计记录 —— "谁批的、批了什么、打回几次"必须留痕，
        # 否则无法回答"这份报告被谁放行的"
        "human": {
            "decision": final.get("human_decision"),
            "rejects": final.get("human_rejects", 0),
            "notes": final.get("human_notes", []),
            "auto_approved": (final.get("human_decision") or "").startswith("approve")
                             and not (final.get("human_notes")),
        },
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


def _print_views(final: dict) -> None:
    """打印 4 份视角结论 + 主管汇总 —— 这是多 Agent 协作最直观的部分。"""
    findings = final.get("findings", [])
    if not findings:
        return
    print(f"\n[视角结论] {len(findings)} 份（并行产出）")
    for f in findings:
        head = (f.get("text") or "").strip().splitlines()
        first = next((l.strip(" -*") for l in head if l.strip()), "")
        print(f"  · {f.get('label', f.get('analyst')):12} 切片={len(f.get('slices_used', []))}片  "
              f"首条：{first[:56]}")
    if final.get("digest"):
        print("\n[主管汇总]")
        for line in (final["digest"] or "").strip().splitlines()[:12]:
            print(f"  {line}")


def _print_review_chain(final: dict) -> None:
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
    # 降级意味着"这一版其实没被审" —— 必须显式告警，不能让它静默通过
    if critique[-1].get("degraded"):
        print("\n  ⚠️ 注意：**最终一次审查降级了**，也就是说最后一版报告"
              "未经实质审查。产物里的 `critique[].degraded` 记录了这一点。")


def _print_run_summary(final: dict, elapsed: float) -> None:
    print("\n[执行轨迹]")
    for step in final.get("trace", []):
        mark = "💰" if step.get("llm_calls") else "🧮"
        extra = {k: v for k, v in step.items() if k not in ("step", "elapsed_s")}
        print(f"  {mark} {step['step']:18} {step['elapsed_s']:>6.3f}s  {extra}")

    _print_views(final)
    _print_review_chain(final)

    u = total_usage(final.get("usages", []))
    print(f"\n[用量] {u['calls']} 次 LLM 调用  model={u['model']}  "
          f"in={u['input_tokens']}  out={u['output_tokens']}  total={u['total_tokens']}")
    if u.get("cache_read"):
        print(f"       prompt 缓存命中 {u['cache_read']} / {u['input_tokens']} 输入 tokens"
              f"（命中率 {u['cache_hit_rate']}%，命中部分通常大幅折扣）")
    print("  按角色：")
    for role, d in u["by_role"].items():
        print(f"    · {role:22} {d['calls']} 次  in={d['input_tokens']:>6}  "
              f"out={d['output_tokens']:>5}  total={d['total_tokens']:>6}")
    cost = estimate_cost(u)
    print("[费用] " + ("未配置单价（在 .env 里填 DEEPSEEK_PRICE_*_PER_M 即可估算）"
                      if cost is None else f"{cost} 元（未计缓存折扣，属保守上界）"))

    # 切片效果：对比"全量事实包"与"各切片之和"
    facts_chars = len(json.dumps(final.get("facts", {}), ensure_ascii=False))
    slice_sum = sum(s.get("slice_chars", 0) for s in final.get("trace", [])
                    if s.get("step", "").startswith("analyst:"))
    if slice_sum:
        print(f"\n[事实包切片] 全量 {facts_chars} 字符；4 份切片合计 {slice_sum} 字符"
              f"（若人手一份全量则为 {facts_chars * len(ROLE_SPECS)} 字符，"
              f"省 {100 - slice_sum * 100 // (facts_chars * len(ROLE_SPECS))}%）")

    print(f"\n[汇总] 工具调用 {sum(s.get('tools_called', 0) for s in final.get('trace', []))} 次，"
          f"LLM 调用 {u['calls']} 次，"
          f"局限 {len(final.get('limitations', []))} 条，总耗时 {elapsed:.2f}s")


def _ask_human(quest: dict) -> str:
    """人工确认关：显示待审批内容并取得答复。

    自动化环境（比如脚本/CI）里没有人可问 —— 这时**自动通过**，
    但必须明确打印告警，**绝不假装"已经人工确认过了"**。
    """
    print("\n" + "=" * 72)
    print("  ⏸  人工确认关 —— 图已暂停（`interrupt()` 生效）")
    print("=" * 72)
    print(f"  待交付报告版本：v{quest.get('revision')}")
    print(f"  上一轮自动审查：{'通过' if quest.get('reviewed_ok') else '未通过或已降级'}"
          f"{'（降级）' if quest.get('reviewed_degraded') else ''}")
    print(f"  审查总评：{str(quest.get('critic_summary', ''))[:88]}")
    print(f"  报告长度：{len(quest.get('report') or '')} 字符（全文见 out/ 产物）")

    if HUMAN_REVIEW and sys.stdin.isatty():
        print("\n  输入决定：`approve` 通过 / `reject:<修改意见>` 打回重写")
        try:
            ans = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = "approve"
        return ans or "approve"

    why = ("HUMAN_REVIEW=0（人工确认已关闭）" if not HUMAN_REVIEW
           else "非交互环境，没有终端可以问人")
    print(f"\n  ⚠️ {why} → **自动通过**。"
          f"这次并没有真正的人工确认，产物里 `human_decision` 会记为 auto-approve。")
    return "approve"


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

    print("=" * 72)
    print(f"  4 分析师扇出 + Supervisor + 审查回路 + 人工确认   复盘日 = {date}")
    print(f"  最多重写 {MAX_REVISIONS - 1} 次 / 人工确认 "
          f"{'开启' if HUMAN_REVIEW else '关闭'}")
    print("=" * 72)
    print("\n[图的 mermaid 定义]\n")
    print(app.get_graph().draw_mermaid())

    t0 = time.time()
    final: dict = {}
    # interrupt 的暂停/恢复必须绑定同一个 thread_id —— 状态靠它找回
    thread = {"configurable": {"thread_id": f"review-{date}"}}
    payload: dict | Command = {"date": date}
    try:
        # 用 stream_mode="values"：每个节点跑完都给一份**完整状态快照**。
        # 好处：后面节点失败时，前面节点（取数/分析师）的成果不会丢。
        for _round in range(MAX_REVISIONS + 2):      # 兜底上限，防人工确认死循环
            interrupted = None
            for chunk in app.stream(payload, config=thread, stream_mode="values"):
                final = chunk
                if chunk.get("__interrupt__"):
                    interrupted = chunk["__interrupt__"]
            if not interrupted:
                break
            # ★ 图在这里暂停，等我们把答复送回去
            item = interrupted[0]
            quest = getattr(item, "value", item)
            payload = Command(resume=_ask_human(quest if isinstance(quest, dict) else {}))
        else:
            print("\n⚠️ 人工确认轮次达到上限，强制收工")
    except Exception as e:
        _diagnose(e)
        if final.get("trace"):
            print("\n   ↳ 已完成节点的成果已保住，不重跑前面的步骤")
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

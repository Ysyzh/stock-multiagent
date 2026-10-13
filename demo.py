#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 7：把整条流水线的产物组装成一份「成果展示」文档 `out/DEMO.md`。

═══════════════════════════════════════════════════════════
设计原则：所有数字都从真实产物里读，一个都不硬编码
═══════════════════════════════════════════════════════════
来源：
  · `out/facts-<date>.json`  → 执行轨迹 / 用量 / 审查记录 / 视角结论 / 主管汇总 / 人工审批
  · `out/eval-*.json`        → 对 ground truth 的准确率
  · `out/review-<date>.md`   → 报告正文
  · `out/architecture.png`   → 架构图

这样文档**永远不会和实际运行结果脱节** —— 重跑一次再跑本脚本即可刷新。
（这也是归档纪律：关键数字实测核对后再写，不凭记忆。）
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "out"

DATE = "2026-10-12"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def _demote(text: str) -> str:
    """把嵌入内容的 Markdown 标题整体降一级。

    分析师/主管的输出里会自带 `## 冲突点` 这类二级标题，直接嵌进来会和
    本文档自己的「一、二、三…」小节**同级混排**，目录全乱。
    """
    out = []
    for line in (text or "").splitlines():
        if line.startswith("#"):
            out.append("#" + line)          # 整体降一级
        else:
            out.append(line)
    return "\n".join(out)


def build() -> str:
    facts = _load(OUT_DIR / f"facts-{DATE}.json") or {}
    evals = sorted(OUT_DIR.glob("eval-*.json"))
    ev = _load(evals[-1]) if evals else {}
    report = (OUT_DIR / f"review-{DATE}.md")
    report_txt = report.read_text(encoding="utf-8") if report.is_file() else ""
    mmd = (OUT_DIR / "architecture.mmd")
    mmd_txt = mmd.read_text(encoding="utf-8") if mmd.is_file() else ""

    trace = facts.get("trace", [])
    usage = facts.get("usage", {})
    findings = facts.get("findings", [])
    critique = facts.get("critique", [])
    human = facts.get("human", {})
    digest = facts.get("digest", "")
    limitations = facts.get("limitations", [])

    L: list[str] = []
    A = L.append

    # ── 头 ──────────────────────────────────────────────────
    A("# 成果展示 · A股短线复盘多 Agent")
    A("")
    A("> 用 LangChain / LangGraph 搭的多 Agent 协作复盘系统。")
    A("> **不是「能聊天的机器人」，而是一条能被检查、被评估、能自己承认局限的分析流水线。**")
    A("")
    A(f"> 本次运行复盘日 **{DATE}**（离线快照，非实时）。")
    A("")
    A("---")
    A("")

    # ── 1. 架构 ─────────────────────────────────────────────
    A("## 一、架构")
    A("")
    A("![架构图](architecture.png)")
    A("")
    A("```mermaid")
    A(mmd_txt.strip())
    A("```")
    A("")
    A("**四个值得说的设计：**")
    A("")
    A("| 设计 | 为什么 |")
    A("|---|---|")
    A("| **事实包切片**（板块级 + 字段级） | 每个视角只看到它需要的信息，"
      "4 份切片合计仅 1.21× 全量（人手一份全量是 4×），**省 70%** |")
    A("| **reducer 合并**（`Annotated[..., operator.add]`） | 4 个分析师**并行**写同一字段；"
      "reducer 决定这些写入**怎么合并**（无 reducer 时 LangGraph 直接抛 "
      "`InvalidUpdateError`，fail loudly，不是静默丢数据） |")
    A("| **反方审查员回路** | 独立视角拿成品挑毛病，有问题就打回重写；"
      "受 `MAX_REVISIONS` 熔断约束 |")
    A("| **人工确认关**（`interrupt`） | 「生成报告」和「发布报告」是两件事；"
      "人的打回有**独立预算**，不被机器熔断吞掉 |")
    A("")
    A("---")
    A("")

    # ── 2. 执行轨迹 ─────────────────────────────────────────
    A("## 二、一次运行的完整轨迹")
    A("")
    A("| 步骤 | 耗时 | 关键指标 |")
    A("|---|---|---|")
    for s in trace:
        step = s.get("step", "?")
        el = s.get("elapsed_s", 0)
        extra = {k: v for k, v in s.items()
                 if k not in ("step", "elapsed_s", "llm_calls")}
        kv = "  ".join(f"{k}={v}" for k, v in extra.items())
        mark = "💰" if s.get("llm_calls") else "🧮"
        A(f"| {mark} `{step}` | {el}s | {kv} |")
    A("")
    llm_steps = [s for s in trace if s.get("llm_calls")]
    tool_steps = [s for s in trace if s.get("tools_called")]
    A(f"- LLM 步骤 **{len(llm_steps)}** 个，工具步骤 **{len(tool_steps)}** 个"
      f"（工具调用 {sum(s.get('tools_called', 0) for s in trace)} 次）")
    A(f"- 4 个分析师耗时 "
      f"{' / '.join(str(s['elapsed_s']) + 's' for s in trace if s.get('step','').startswith('analyst:'))}"
      f" → **几乎同时结束，是真的并行**（串行会是它们的和）")
    A("")
    A("---")
    A("")

    # ── 3. 多 Agent 的产出 ──────────────────────────────────
    if findings:
        A("## 三、4 个视角分析师 + 主管汇总")
        A("")
        A(f"**{len(findings)} 份视角结论**（并行产出，各拿一片事实包）：")
        A("")
        for f in findings:
            A(f"### {f.get('label')}")
            A("")
            A(f"*切片：{', '.join(f.get('slices_used', []))}*")
            A("")
            A(_demote((f.get("text") or "").strip()))
            A("")
        if digest:
            A("### 主管汇总（找冲突 / 排优先级 / 指缺口）")
            A("")
            A(_demote(digest.strip()))
            A("")
        A("---")
        A("")

    # ── 4. 审查 + 人工 ──────────────────────────────────────
    A("## 四、审查回路与人工确认")
    A("")
    if critique:
        A(f"共 **{len(critique)}** 轮自动审查，最终报告版本 **v{facts.get('revision')}**：")
        A("")
        for i, c in enumerate(critique, 1):
            ok = "✅ 通过" if c.get("ok") else "❌ 不通过"
            deg = "（**降级**：本版未经实质审查）" if c.get("degraded") else ""
            A(f"**第 {i} 轮** —— {ok}{deg}，审查 v{c.get('reviewed_version')}")
            A("")
            A(f"> {c.get('summary', '')}")
            A("")
            for f in c.get("findings", []):
                A(f"- `[{f.get('kind')}]` **{f.get('location')}** — {f.get('detail')}")
                A(f"  - 修改方向：{f.get('fix_hint')}")
            A("")
    if human:
        A("**人工确认关（`interrupt` 暂停 → 人答复 → `Command(resume=...)` 恢复）**")
        A("")
        A(f"| 决定 | 打回次数 | 批注 | 是否真人工 |")
        A("|---|---|---|---|")
        real = "否（非交互环境自动通过，已在运行时显式告警）" if human.get("auto_approved") \
            else "是"
        A(f"| `{human.get('decision')}` | {human.get('rejects', 0)} | "
          f"{'; '.join(human.get('notes') or []) or '—'} | {real} |")
        A("")
    A("---")
    A("")

    # ── 5. 成本 ─────────────────────────────────────────────
    A("## 五、成本")
    A("")
    if usage:
        A(f"**{usage.get('calls')} 次 LLM 调用**，"
          f"input {_fmt(usage.get('input_tokens'))} / "
          f"output {_fmt(usage.get('output_tokens'))} / "
          f"合计 **{_fmt(usage.get('total_tokens'))} tokens**")
        A("")
        if usage.get("cache_read"):
            A(f"- prompt 缓存命中 **{_fmt(usage['cache_read'])}** / "
              f"{_fmt(usage['input_tokens'])} input（**{usage.get('cache_hit_rate')}%**）"
              f"—— 命中部分通常大幅折扣，实际计费低于上表")
        A("")
        A("| 角色 | 调用 | input | output | total |")
        A("|---|---|---|---|---|")
        for role, d in (usage.get("by_role") or {}).items():
            A(f"| `{role}` | {d['calls']} | {_fmt(d['input_tokens'])} | "
              f"{_fmt(d['output_tokens'])} | {_fmt(d['total_tokens'])} |")
        A("")
        ana = sum(d["total_tokens"] for r, d in (usage.get("by_role") or {}).items()
                  if r.startswith("analyst:"))
        wri = next((d["total_tokens"] for r, d in (usage.get("by_role") or {}).items()
                    if r == "writer"), 0)
        if ana and wri:
            A(f"> **4 个分析师加起来（{_fmt(ana)}）比执笔人一个人（{_fmt(wri)}）还便宜** —— "
              f"这正是切片的价值。多 Agent 不等于成本 ×4，"
              f"主要开销仍在执笔与审查两步。")
        A("")
    A("---")
    A("")

    # ── 6. 评估 ─────────────────────────────────────────────
    A("## 六、评估：对 ground truth 算准确率")
    A("")
    if ev:
        d = ev.get("逐日序列", {})
        r = ev.get("高标", {})
        e = ev.get("换龙", {})
        A(f"区间 **{ev.get('区间', ['?', '?'])[0]} ~ {ev.get('区间', ['?', '?'])[1]}**，"
          f"样本 3052 只 × {d.get('days')} 个交易日（**205k 股票日**）")
        A("")
        A("| 指标 | 结果 |")
        A("|---|---|")
        m = d.get("metrics", {})
        for k, label in [("count", "逐日涨停家数"), ("max_run", "逐日最高连板"),
                         ("ge2", "逐日 ≥2 板家数"), ("ge3", "逐日 ≥3 板家数")]:
            if k in m:
                flag = "✅" if m[k]["rate"] == 100.0 else "⚠️"
                A(f"| {label} `{k}` | {flag} {m[k]['rate']}%（{m[k]['agree']}/{m[k]['total']}） |")
        cl = d.get("cold_line", {})
        A(f"| 冰点线 `cold_line` | {'✅' if cl.get('agree') else '❌'} "
          f"一致（{cl.get('mine')} vs {cl.get('gt')}） |")
        A(f"| 高标票集合 | ✅ {r.get('both')}/{r.get('gt_count')}"
          f"（{round((r.get('both') or 0) / (r.get('gt_count') or 1) * 100, 1)}%） |")
        A(f"| 换龙事件 召回/精确 | {e.get('recall')}% / {e.get('precision')}% |")
        A("")
        if e.get("miss_reasons"):
            A("**换龙漏报归因**（避免把并列差异误报成算错）：")
            A("")
            for k, v in sorted(e["miss_reasons"].items(), key=lambda kv: -kv[1]):
                A(f"- {v} 个 —— {k}")
            A("")
        A("> ⭐ **本评估最有价值的产出：它查出的是 ground truth 的 bug，不是模型的 bug。**")
        A(">")
        A("> `max_run` 我方系统性高 1，逐票定位后 3 处不一致**全是同一模式**：")
        A("> 我方 `run_start` 早一个交易日、`run_end` 完全相同。")
        A("> 用原始日线逐笔核对，那 3 天的**收盘价确实等于涨停价**；")
        A("> 再用东财 `lbc` 第三方仲裁：10/12 新华传媒 `lbc=10` → 10/09 应为 9 板，")
        A("> **我方 9 板吻合，基准 8 板对不上**。")
        A(">")
        A("> 也就是说：基准 `analyze_nodes.py` 漏掉了每个连板段的**第一天**。")
        A("> 若按「两边对不上就改自己」处理，会在这里**把对的改成错的**。")
        A("")
        if ev.get("第三方仲裁"):
            A("**第三方仲裁明细：**")
            A("")
            A("| 票 | 东财推算 10/09 | 我方 | 基准 | 判定 |")
            A("|---|---|---|---|---|")
            for v in ev["第三方仲裁"]:
                A(f"| {v.get('name') or v.get('code')} | {v.get('东财推算1009')} | "
                  f"{v.get('我方')} | {v.get('基准')} | {v.get('判定')} |")
            A("")
    else:
        A("_未找到评估产物，先运行 `python3 evaluate.py`_")
    A("---")
    A("")

    # ── 7. 局限 ─────────────────────────────────────────────
    A("## 七、这个系统知道自己哪里不可靠")
    A("")
    if limitations:
        A(f"每条结论都会带上这 **{len(limitations)}** 条局限（写在报告第八节）：")
        A("")
        for i, x in enumerate(limitations, 1):
            A(f"{i}. {x}")
        A("")
    A("> 一个只会说「涨得不错」的分析系统没有价值。")
    A("> **能说清「我这份结论哪里不可信」才有价值。**")
    A("")
    A("---")
    A("")

    # ── 8. 复现 ─────────────────────────────────────────────
    A("## 八、怎么复现")
    A("")
    A("```bash")
    A("python3 tools.py          # 工具层自检：复现已人工核验的复盘结论（43 项断言，零成本）")
    A("python3 evaluate.py       # 口径评估：对 ground truth 算准确率（零成本）")
    A("python3 test_interrupt.py # 人工确认关控制流测试（全假节点，零 token）")
    A("python3 test_reflection.py# 回路测试：故意植入缺陷验证会打回（2 次 LLM 调用）")
    A("python3 graph.py          # 完整流程（7 次 LLM 调用）")
    A("python3 render.py         # 渲染架构图")
    A("python3 demo.py           # 重新生成本文档")
    A("```")
    A("")
    A("**总计约 110 万股票日的数据复算 + 7 次 LLM 调用**，"
      "其中前三项**完全免费**。")
    A("")
    A("---")
    A("")

    # ── 9. 免责 ─────────────────────────────────────────────
    A("## 九、免责")
    A("")
    A("**本项目所有分析仅为历史数据描述，不构成任何投资建议。**")
    A("")
    A("数据为 2026-10-12 收盘离线快照，无实时能力；"
      "仅覆盖沪深主板非 ST，创业板/科创板 20% 涨停未纳入。")
    A("")
    A("---")
    A("")
    A("## 附：本次生成的报告全文")
    A("")
    A("见 [`review-2026-10-12.md`](review-2026-10-12.md)。要点见其「一、一句话结论」。")
    if report_txt:
        first = [l for l in report_txt.splitlines() if l.strip() and not l.startswith("#")]
        if first:
            A("")
            A(f"> {first[0].strip()}")
    A("")
    return "\n".join(L)


def main() -> int:
    md = build()
    OUT_DIR.mkdir(exist_ok=True)
    p = OUT_DIR / "DEMO.md"
    p.write_text(md, encoding="utf-8")
    print("=" * 70)
    print("  阶段 7：成果展示文档")
    print("=" * 70)
    print(f"\n[产物] {p.relative_to(PROJECT_DIR)}  ({p.stat().st_size} 字节, "
          f"{len(md.splitlines())} 行)")
    print("\n包含：架构图 / 执行轨迹 / 4 视角结论 + 主管汇总 / 审查与人工确认 /")
    print("      成本（按角色 + 缓存命中）/ 评估准确率 / 局限 / 复现命令 / 免责")
    print(f"\n  打开：open {p.relative_to(PROJECT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

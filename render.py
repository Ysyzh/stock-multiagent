#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 7：把架构图从 mermaid 源码渲染成「能拿给人看」的东西。

三种输出，**逐级降级，前面的失败不影响后面的**：

| 输出 | 依赖 | 说明 |
|---|---|---|
| `out/architecture.html` | 无（mermaid 从 CDN 加载） | **一定能生成**，双击浏览器打开即可 |
| `out/architecture.png` | 网络（mermaid.ink） | 失败就跳过，附上原因 |
| `out/architecture.svg` | 本机装了 `mmdc` | 当前环境未装，存在就用 |

> 为什么不直接 `draw_mermaid_png()`：它把图交给第三方服务 mermaid.ink 渲染，
> 需要联网。**演示时最不能依赖的就是网络**，所以主输出是自包含 HTML。

运行：python3 render.py
"""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graph as G

PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "out"


# ════════════════════════════════════════════════════════════
#  1. mermaid 源码
# ════════════════════════════════════════════════════════════

def grab_mermaid() -> str:
    """现取一份最新的图定义（比读旧文件可靠 —— 图改了这里就跟着变）。"""
    app = G.build_graph()
    src = app.get_graph().draw_mermaid()
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "architecture.mmd").write_text(src, encoding="utf-8")
    return src


def _strip_frontmatter(src: str) -> str:
    """langgraph 会在 mermaid 前面加一段 `--- config: ... ---` 的 frontmatter。

    CDN 版的 mermaid 对这段很挑（版本/语法稍不对就整图不渲染），
    所以做一张**去 frontmatter** 的副本给 HTML 用 —— 降低"演示时白屏"的风险。
    """
    lines = src.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                rest = lines[i + 1:]
                # 保留真正画图的 `flowchart/graph` 行之后的内容
                for j, l in enumerate(rest):
                    if l.strip().startswith(("graph ", "flowchart ")):
                        return "\n".join(rest[j:])
                return "\n".join(rest)
    return src


# ════════════════════════════════════════════════════════════
#  2. 自包含 HTML（主输出）
# ════════════════════════════════════════════════════════════

_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>stock-multiagent 架构图</title>
<style>
  body {{ margin:0; padding:28px 22px; font-family:-apple-system,"PingFang SC",sans-serif;
          background:#faf9fc; color:#222; }}
  h1 {{ font-size:19px; margin:0 0 4px; }}
  .sub {{ color:#777; font-size:13px; margin-bottom:20px; }}
  #box {{ background:#fff; border:1px solid #e6e2f0; border-radius:10px; padding:22px;
          box-shadow:0 1px 4px rgba(90,70,150,.06); overflow:auto; }}
  details {{ margin-top:18px; }}
  summary {{ cursor:pointer; color:#5a4a96; font-size:13px; }}
  pre {{ background:#f4f2fa; padding:12px; border-radius:8px; font-size:12px; overflow:auto; }}
  .err {{ color:#b00; font-size:13px; }}
</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="sub">{sub}</div>
  <div id="box"><pre class="mermaid">{src}</pre></div>

  <details>
    <summary>查看 mermaid 源码</summary>
    <pre>{src_raw}</pre>
  </details>

  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'base',
      themeVariables: {{ primaryColor:'#efeaff', primaryBorderColor:'#9c8fd0',
                         lineColor:'#7b6bb5', fontFamily:'-apple-system,PingFang SC,sans-serif' }} }});
  </script>
</body>
</html>
"""


def write_html(src: str, title: str, sub: str) -> Path:
    clean = _strip_frontmatter(src)
    p = OUT_DIR / "architecture.html"
    p.write_text(_HTML.format(title=html.escape(title), sub=html.escape(sub),
                              src=html.escape(clean), src_raw=html.escape(src)),
                 encoding="utf-8")
    return p


# ════════════════════════════════════════════════════════════
#  3. PNG（联网，可选）
# ════════════════════════════════════════════════════════════

def try_png() -> tuple[Path | None, str]:
    """调 langgraph 的 draw_mermaid_png（走 mermaid.ink）。失败不算错。"""
    try:
        app = G.build_graph()
        data = app.get_graph().draw_mermaid_png()
        p = OUT_DIR / "architecture.png"
        p.write_bytes(data)
        return p, f"成功（{len(data)} 字节）"
    except Exception as e:
        return None, f"跳过：{type(e).__name__}: {str(e)[:110]}"


# ════════════════════════════════════════════════════════════
#  4. SVG（本机 mmdc，可选）
# ════════════════════════════════════════════════════════════

def try_svg() -> tuple[Path | None, str]:
    if not shutil.which("mmdc"):
        return None, "跳过：本机未安装 mmdc（mermaid-cli）"
    try:
        subprocess.run(
            ["mmdc", "-i", str(OUT_DIR / "architecture.mmd"),
             "-o", str(OUT_DIR / "architecture.svg"), "-b", "transparent"],
            check=True, capture_output=True, timeout=120, cwd=PROJECT_DIR,
        )
        return OUT_DIR / "architecture.svg", "成功"
    except Exception as e:
        return None, f"跳过：{type(e).__name__}: {str(e)[:110]}"


def main() -> int:
    print("=" * 70)
    print("  阶段 7：渲染架构图")
    print("=" * 70)

    src = grab_mermaid()
    nodes = [l.strip() for l in src.splitlines() if "-->" in l or "-." in l]
    print(f"\n[图定义] {len(src)} 字符，{len(nodes)} 条边")

    p = write_html(src, "stock-multiagent · 多 Agent 复盘流水线",
                   "4 视角分析师并行扇出 → Supervisor 汇总 → 执笔 → 反方审查 → "
                   "人工确认；含打回重写回路与熔断")
    print(f"\n[主输出] {p.relative_to(PROJECT_DIR)}  ({p.stat().st_size} 字节)")
    print("         ✅ 自包含 HTML，双击浏览器打开（mermaid 走 CDN）")

    png, msg = try_png()
    print(f"[PNG]    {'✅ ' + str(png.relative_to(PROJECT_DIR)) if png else '⚠️  ' + msg}")

    svg, msg2 = try_svg()
    print(f"[SVG]    {'✅ ' + str(svg.relative_to(PROJECT_DIR)) if svg else '⚠️  ' + msg2}")

    print("\n[如何使用]")
    print(f"  open {p.relative_to(PROJECT_DIR)}      # macOS 直接打开架构图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""extract-ima-html.py — 从 ima.copilot「完整网页导出」HTML 抽 Markdown 正本。

用法:
    python3 bin/extract-ima-html.py <input.html> <output.md> [--title "..."] [--source "..."]

只依赖标准库（html.parser）。识别 ima 导出的两类消息节点:
    用户: class 含 `_chatMainBubble_`
    ima : class 含 `_markdown_`
按文档顺序抽出，AI 侧 HTML(含 hljs 高亮 / 表格 / 列表) → Markdown。
"""
import sys
import re
import html as _html
from html.parser import HTMLParser

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}
SKIP_CLASS = ("take-note-exclude", "copy-exclude", "_toolbar", "_operation",
              "_icon", "_header_")
SKIP_TAG = {"script", "style", "svg", "path", "button", "noscript", "template"}


class Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag, attrs=None, parent=None):
        self.tag = tag
        self.attrs = attrs or {}
        self.children = []
        self.parent = parent


class Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {k.lower(): (v or "") for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        node = Node(tag, {k.lower(): (v or "") for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def cls(node):
    return node.attrs.get("class", "")


def has(node, needle):
    return needle.lower() in cls(node).lower()


def skip(node):
    if isinstance(node, str):
        return False
    if node.tag in SKIP_TAG:
        return True
    c = cls(node).lower()
    return any(s in c for s in SKIP_CLASS)


def text_of(node):
    if isinstance(node, str):
        return node
    if skip(node):
        return ""
    return "".join(text_of(c) for c in node.children)


def iter_all(node):
    for c in node.children:
        if isinstance(c, str):
            continue
        yield c
        yield from iter_all(c)


def children_md(node):
    return "".join(_to_md(c) for c in node.children)


def _to_md(node):
    if isinstance(node, str):
        return node
    if skip(node):
        return ""
    t = node.tag
    if t in ("strong", "b"):
        return "**" + children_md(node).strip() + "**"
    if t in ("em", "i"):
        return "*" + children_md(node).strip() + "*"
    if t in ("del", "s", "strike"):
        return "~~" + children_md(node).strip() + "~~"
    if t == "br":
        return "\n"
    if t == "hr":
        return "\n\n---\n\n"
    if t in ("h1", "h2", "h3", "h4", "h5", "h6"):
        n = min(int(t[1]) + 1, 6)  # 降一级: 消息层用 ##，正文标题退到 ### 以下
        return "\n\n" + "#" * n + " " + children_md(node).strip() + "\n\n"
    if t == "p":
        return "\n\n" + children_md(node).strip() + "\n\n"
    if t == "blockquote":
        body = children_md(node).strip()
        return "\n\n" + "\n".join("> " + ln for ln in body.splitlines()) + "\n\n"
    if t == "code":
        if node.parent is not None and node.parent.tag == "pre":
            return text_of(node)
        return "`" + text_of(node).strip() + "`"
    if t == "pre":
        code = "\n".join(text_of(node).strip("\n").splitlines())
        return "\n\n```\n" + code + "\n```\n\n"
    if t == "a":
        href = node.attrs.get("href", "")
        txt = children_md(node).strip()
        if not href:
            return txt
        return "[" + (txt if txt else href) + "](" + href + ")"
    if t == "img":
        src = node.attrs.get("src", "")
        alt = node.attrs.get("alt", "")
        if src.startswith("./"):
            src = src[2:]
        return "\n\n![" + alt + "](" + src + ")\n\n"
    if t in ("ul", "ol"):
        return _list_md(node)
    if t == "li":
        return children_md(node).strip()
    if t == "table":
        return _table_md(node)
    if t == "katex" or has(node, "katex"):
        ann = next((x for x in iter_all(node) if x.tag == "annotation"
                    and "tex" in x.attrs.get("encoding", "")), None)
        if ann is not None:
            return "$" + text_of(ann).strip() + "$"
    # 结构容器: 直接递归
    return children_md(node)


def _list_md(node):
    ordered = node.tag == "ol"
    lines = []
    idx = 1
    for li in node.children:
        if isinstance(li, str) or li.tag != "li":
            continue
        body = children_md(li).strip()
        body = body.replace("\n\n", "\n").strip()
        # 嵌套列表: 次级行缩进
        parts = body.splitlines()
        prefix = (str(idx) + ". ") if ordered else "- "
        lines.append(prefix + (parts[0] if parts else ""))
        for p in parts[1:]:
            lines.append("   " + p)
        idx += 1
    return "\n\n" + "\n".join(lines) + "\n\n"


def _table_md(node):
    rows = []
    for tr in iter_all(node):
        if tr.tag != "tr":
            continue
        cells = [text_of(td).strip().replace("\n", " ").replace("|", "\\|")
                 for td in tr.children if not isinstance(td, str) and td.tag in ("td", "th")]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |",
           "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n\n" + "\n".join(out) + "\n\n"


def extract(html_text):
    p = Builder()
    p.feed(html_text)
    msgs = []  # (role, node) 文档序
    def walk(node):
        for c in iter_all(node):
            if c.tag == "div" and "_chatmainbubble_" in cls(c).lower():
                msgs.append(("用户", c))
            elif "_markdown_" in cls(c).lower() and not has(c, "_chatmainbubble_"):
                msgs.append(("ima.copilot", c))
    walk(p.root)
    return msgs


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    title = "ima.copilot 会话导出"
    source = ""
    args = sys.argv[3:]
    for i, a in enumerate(args):
        if a == "--title" and i + 1 < len(args):
            title = args[i + 1]
        if a == "--source" and i + 1 < len(args):
            source = args[i + 1]
    raw = open(src, encoding="utf-8", errors="replace").read()
    msgs = extract(raw)
    parts = [f"# {title}\n"]
    if source:
        parts.append(f"> 源：`{source}`（完整网页导出 · DOM 保真）")
    parts.append("> 本件由 `bin/extract-ima-html.py` 从该 HTML 提取（Markdown 正本，可入仓）\n")
    parts.append("---\n")
    turn = 0
    i = 0
    users = ais = 0
    total_nodes = len(msgs)
    while i < total_nodes:
        role, node = msgs[i]
        if role == "用户":
            turn += 1
            users += 1
            body = re.sub(r"\n{3,}", "\n\n", text_of(node).strip())
            parts.append(f"## {turn} · 用户\n\n{body}\n")
            i += 1
            continue
        # 合并相邻的 ima 块（同一轮答案可能被拆成多个 _markdown_ 容器）
        chunks = []
        while i < total_nodes and msgs[i][0] != "用户":
            chunks.append(_to_md(msgs[i][1]).strip())
            i += 1
        ais += 1
        body = re.sub(r"\n{3,}", "\n\n", "\n\n".join(c for c in chunks if c))
        parts.append(f"## {turn} · ima.copilot\n\n{body}\n")
    open(dst, "w", encoding="utf-8").write("\n".join(parts).rstrip() + "\n")
    print(f"extracted: {total_nodes} nodes -> {turn} turns ({users} user / {ais} ima) -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

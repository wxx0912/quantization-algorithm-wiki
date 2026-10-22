"""知识页的严格读取、检索与内容结构检查；不判断算法结论是否正确。"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
TYPES = {"algorithm", "foundation", "guide", "recipe", "case"}
FIELDS = {"id", "title", "type", "tags", "aliases", "related", "sources", "updated"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 重复键: {key}")
        result[key] = value
    return result


def pages(root=ROOT):
    result = []
    for path in sorted((root / "wiki").rglob("*.md")):
        if path.name == "TEMPLATE.md":
            continue
        text = path.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            raise ValueError(f"{path}: 缺少 frontmatter 分隔符")
        meta = json.loads(parts[1], object_pairs_hook=unique_object)
        if not isinstance(meta, dict):
            raise ValueError(f"{path}: frontmatter 必须为 JSON 对象")
        result.append((path, meta, parts[2].strip()))
    return result


def sources(root=ROOT):
    result = {}
    for path in sorted((root / "sources").glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| `"):
                continue
            match = re.fullmatch(r"\| `([^`]+)` \| ([^|]+) \| (https?://\S+) \| (.+) \|", line)
            if not match:
                raise ValueError(f"{path}: 非法来源行 {line}")
            source_id, kind, url, locator = match.groups()
            if source_id in result:
                raise ValueError(f"重复来源 ID: {source_id}")
            result[source_id] = {"type": kind, "url": url, "locator": locator}
    return result


def search(records, query="", tag=None, kind=None, limit=10):
    terms = query.casefold().split()
    results = []
    for path, meta, body in records:
        if tag and tag not in meta["tags"]:
            continue
        if kind and kind != meta["type"]:
            continue
        names = [meta["id"], meta["title"], *meta["aliases"]]
        name_text = " ".join(names).casefold()
        tags = " ".join(meta["tags"]).casefold()
        haystack = " ".join([name_text, tags, body.casefold()])
        if not all(term in haystack for term in terms):
            continue
        exact = bool(query) and query.casefold() in [n.casefold() for n in names]
        score = 100 * exact + sum(10 * (t in name_text) + 3 * (t in tags) + (t in haystack) for t in terms)
        results.append((score, path, meta))
    results.sort(key=lambda row: (-row[0], row[2]["id"]))
    return results[:limit]


def validate(root=ROOT):
    records = pages(root)
    registry = sources(root)
    errors = []
    ids = set()
    for path, meta, body in records:
        label = str(path.relative_to(root))
        if set(meta) != FIELDS:
            errors.append(f"{label}: 字段不匹配 {sorted(set(meta) ^ FIELDS)}")
        page_id = meta.get("id")
        if not isinstance(page_id, str) or not re.fullmatch(r"[a-z][a-z0-9-]+", page_id):
            errors.append(f"{label}: 非法 ID")
        elif page_id in ids:
            errors.append(f"{label}: 重复 ID {page_id}")
        else:
            ids.add(page_id)
        if not isinstance(meta.get("type"), str) or meta["type"] not in TYPES:
            errors.append(f"{label}: 未知页面类型")
        if not isinstance(meta.get("title"), str) or not meta.get("title"):
            errors.append(f"{label}: 缺少标题")
        for field in ("tags", "aliases", "related", "sources"):
            value = meta.get(field)
            if not isinstance(value, list) or not all(isinstance(x, str) and x for x in value):
                errors.append(f"{label}: {field} 必须为非空字符串组成的数组")
            elif len(set(value)) != len(value):
                errors.append(f"{label}: {field} 存在重复项")
        try:
            updated = meta.get("updated", "")
            if not isinstance(updated, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", updated):
                raise ValueError("日期应为 YYYY-MM-DD")
            date.fromisoformat(updated)
        except (TypeError, ValueError):
            errors.append(f"{label}: 日期非法")
        if not body.startswith("# "):
            errors.append(f"{label}: 正文缺少一级标题")
        if not meta.get("sources"):
            errors.append(f"{label}: 缺少来源")
    for path, meta, _ in records:
        for field, targets in (("related", ids), ("sources", registry)):
            value = meta.get(field)
            if isinstance(value, list):
                for target in value:
                    if isinstance(target, str) and target not in targets:
                        errors.append(f"{path.relative_to(root)}: {field} 未登记 {target}")
    for path in sorted(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for angle, plain in re.findall(r"\]\((?:<([^>]+)>|([^\s)]+))(?:\s+\"[^\"]*\")?\)", text):
            destination = angle or plain
            if re.match(r"[a-zA-Z]+:", destination):
                continue
            target, _, fragment = destination.partition("#")
            resolved = path.parent / unquote(target) if target else path
            if not resolved.exists():
                errors.append(f"{path.relative_to(root)}: 失效本地链接 {destination}")
            elif fragment and resolved.suffix == ".md":
                target_text = re.sub(r"```.*?```", "", resolved.read_text(encoding="utf-8"), flags=re.S)
                anchors = set()
                counts = {}
                for title in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", target_text, flags=re.M):
                    slug = re.sub(r"[^\w\-\s]", "", title.lower()).replace(" ", "-")
                    count = counts.get(slug, 0)
                    anchors.add(f"{slug}-{count}" if count else slug)
                    counts[slug] = count + 1
                if unquote(fragment) not in anchors:
                    errors.append(f"{path.relative_to(root)}: 失效标题锚点 {destination}")
    return errors, len(records), len(registry)

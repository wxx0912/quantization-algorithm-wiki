"""由页面元数据生成索引，正文内容不复制到索引。"""
import argparse
from wiki import ROOT, pages


def render():
    records = pages()
    lines = ["# 页面索引", "", "> 由 `python3 scripts/build_index.py` 生成，请勿手改。", ""]
    for kind, label in (("foundation", "基础"), ("algorithm", "算法"), ("guide", "指南"), ("recipe", "Recipe"), ("case", "案例")):
        lines.extend([f"## {label}", ""])
        for path, meta, _ in sorted(records, key=lambda r: r[1]["id"]):
            if meta["type"] == kind:
                lines.append(f"- `{meta['id']}` — [{meta['title']}]({path.relative_to(ROOT).as_posix()})")
        lines.append("")
    lines.extend(["## 按机制标签", ""])
    for tag in sorted({tag for _, meta, _ in records for tag in meta["tags"]}):
        links = [f"[{meta['title']}]({path.relative_to(ROOT).as_posix()})" for path, meta, _ in records if tag in meta["tags"]]
        lines.append(f"- `{tag}`：" + "；".join(links))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "index.generated.md"
    content = render()
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            parser.exit(1, "索引缺失或过期，请运行 scripts/build_index.py\n")
        print("索引与页面元数据一致")
    else:
        path.write_text(content, encoding="utf-8")

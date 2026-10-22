"""按名称、别名、机制标签或问题关键词检索。"""
import argparse
import json
from wiki import ROOT, TYPES, pages, search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--tag")
    parser.add_argument("--type", choices=sorted(TYPES))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 必须大于零")
    rows = [{"id": meta["id"], "title": meta["title"], "path": str(path.relative_to(ROOT)),
             "type": meta["type"], "tags": meta["tags"], "sources": meta["sources"]}
            for _, path, meta in search(pages(), args.query, args.tag, args.type, args.limit)]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"{row['id']}\t{row['title']}\t{row['path']}")
        if not rows:
            print("未命中；尝试算法别名、较短的问题关键词或 --tag。未命中不代表方法不可用。")


if __name__ == "__main__":
    main()

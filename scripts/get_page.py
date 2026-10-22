"""按稳定 ID 读取一个页面，可同时展开来源登记。"""
import argparse
import json
from wiki import pages, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("id")
    parser.add_argument("--follow-sources", action="store_true")
    args = parser.parse_args()
    found = [(path, meta) for path, meta, _ in pages() if meta["id"] == args.id]
    if len(found) != 1:
        parser.error(f"页面 ID 应唯一命中，实际 {len(found)} 个: {args.id}")
    path, meta = found[0]
    print(path.read_text(encoding="utf-8"))
    if args.follow_sources:
        registry = sources()
        print(json.dumps({key: registry[key] for key in meta["sources"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

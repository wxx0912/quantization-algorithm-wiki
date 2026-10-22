"""检查元数据、来源 ID、关联页和本地文件链接。"""
import sys
from wiki import validate


if __name__ == "__main__":
    errors, page_count, source_count = validate()
    for error in errors:
        print(error, file=sys.stderr)
    print(f"{page_count} pages, {source_count} sources, {len(errors)} errors")
    sys.exit(bool(errors))

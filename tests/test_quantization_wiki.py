"""检查本库知识内容与数据层的一致性；不验证共享工具的行为（见 test_shared_tools.py）。"""
from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from wiki_common import alias_pages, pages, search, sources, validate, vocabulary


class QuantizationWikiContentTest(unittest.TestCase):
    def test_repository_validates_without_errors_or_warnings(self):
        errors, warnings, page_count, source_count = validate(ROOT)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(page_count, 20)
        self.assertEqual(source_count, 32)

    def test_alias_keys_resolve_to_registered_pages(self):
        vocab = vocabulary(ROOT)
        known = {page.meta["id"] for page in pages(ROOT)}
        expansions = vocab["alias_expansions"]
        for key, targets in expansions.items():
            with self.subTest(alias=key):
                self.assertTrue(key.strip())
                self.assertTrue(targets)
                self.assertEqual(targets, list(dict.fromkeys(targets)), "别名目标不得重复")
                self.assertEqual(set(targets) - known, set())
                hits = [row[1].meta["id"] for row in search(pages(ROOT), ids=alias_pages(vocab, key))]
                self.assertEqual(hits, list(dict.fromkeys(targets)))

    def test_every_declared_tag_is_used_by_a_page(self):
        declared = [tag["name"] if isinstance(tag, dict) else tag for tag in vocabulary(ROOT)["tags"]]
        used = {tag for page in pages(ROOT) for tag in page.meta["tags"]}
        self.assertEqual(sorted(set(declared) - used), [])

    def test_every_topic_is_declared_and_populated(self):
        declared = {topic["name"] for topic in vocabulary(ROOT)["source_topics"]}
        used = {source.meta["topic"] for source in sources(ROOT).values()}
        self.assertEqual(used, declared)
        for topic in (ROOT / "sources" / "topics").glob("*.md"):
            self.assertIn(topic.stem, declared)

    def test_migration_audit_covers_every_source_exactly_once(self):
        audit = (ROOT / "sources" / "MIGRATION-AUDIT.md").read_text(encoding="utf-8")
        lookup = re.findall(r"^\| ([a-z][a-z0-9-]+) \| \d{4}-", audit, flags=re.M)
        self.assertEqual(len(lookup), len(set(lookup)))
        self.assertEqual(set(lookup), set(sources(ROOT)))

    def test_generated_indices_are_current(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_index.py"), "--check"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_source_is_referenced_by_a_page_and_indexed(self):
        registry = set(sources(ROOT))
        referenced = {source_id for page in pages(ROOT) for source_id in page.meta["sources"]}
        self.assertEqual(registry - referenced, set())
        index = (ROOT / "sources" / "INDEX.generated.md").read_text(encoding="utf-8")
        missing = [source_id for source_id in registry if f"`{source_id}`" not in index]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()

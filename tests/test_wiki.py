"""用独立临时知识库检查检索、数据约束与 CLI；不验证算法内容。"""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location("wiki_under_test", SCRIPTS / "wiki.py")
wiki = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wiki)


class WikiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dev_v0_1_20261023_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repository"
        (self.root / "wiki").mkdir(parents=True)
        (self.root / "sources").mkdir()
        shutil.copytree(SCRIPTS, self.root / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
        self.outside = Path(self.temp.name) / "outside"
        self.outside.mkdir()
        self.source_file = self.root / "sources" / "papers.md"
        self.source_file.write_text(
            "| ID | 类型 | URL | 版本与已读定位 |\n"
            "|---|---|---|---|\n"
            "| `paper-example` | 论文 | https://example.com/paper | v1 §3 |\n",
            encoding="utf-8",
        )

    def page(self, filename="example", body="# 示例\n\n正文。", **changes):
        meta = {
            "id": "algorithm-example", "title": "示例方法", "type": "algorithm",
            "tags": ["重构"], "aliases": ["Example"], "related": [],
            "sources": ["paper-example"], "updated": "2026-10-23",
        }
        meta.update(changes)
        path = self.root / "wiki" / f"{filename}.md"
        path.write_text("---\n" + json.dumps(meta, ensure_ascii=False) + "\n---\n" + body + "\n", encoding="utf-8")
        return path

    def cli(self, script, *args):
        return subprocess.run(
            [sys.executable, str(self.root / "scripts" / script), *args],
            cwd=self.outside, text=True, capture_output=True, check=False,
        )

    def errors(self):
        return wiki.validate(self.root)[0]

    def test_valid_fixture(self):
        self.page()
        self.assertEqual(wiki.validate(self.root), ([], 1, 1))

    def test_exact_alias_ranks_before_partial_title_and_body(self):
        self.page("a", id="algorithm-a", title="GPTQ notes", aliases=[], body="# A\nGPTQ GPTQ GPTQ")
        self.page("z", id="algorithm-z", title="目标", aliases=["GPTQ"])
        rows = wiki.search(wiki.pages(self.root), "gPtQ")
        self.assertEqual([row[2]["id"] for row in rows], ["algorithm-z", "algorithm-a"])

    def test_query_uses_and_and_combines_tag_type_filters(self):
        self.page("both", id="algorithm-both", tags=["缩放"], body="# A\nactivation calibration")
        self.page("partial", id="algorithm-partial", tags=["缩放"], body="# B\nactivation")
        self.page("guide", id="guide-both", type="guide", tags=["缩放"], body="# C\nactivation calibration")
        self.page("other", id="algorithm-other", tags=["舍入"], body="# D\nactivation calibration")
        rows = wiki.search(wiki.pages(self.root), "activation calibration", tag="缩放", kind="algorithm")
        self.assertEqual([row[2]["id"] for row in rows], ["algorithm-both"])

    def test_empty_results_have_json_contract(self):
        self.page()
        self.assertEqual(wiki.search(wiki.pages(self.root), "not-present"), [])
        result = self.cli("query.py", "not-present", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_cli_is_independent_of_working_directory(self):
        self.page()
        result = self.cli("query.py", "Example", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0]["path"], "wiki/example.md")
        page = self.cli("get_page.py", "algorithm-example", "--follow-sources")
        self.assertEqual(page.returncode, 0, page.stderr)
        self.assertIn("https://example.com/paper", page.stdout)
        validation = self.cli("validate.py")
        self.assertEqual(validation.returncode, 0, validation.stderr)

    def test_duplicate_page_id_is_rejected(self):
        self.page("first")
        self.page("second")
        self.assertTrue(any("重复 ID" in error for error in self.errors()))
        result = self.cli("get_page.py", "algorithm-example")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("实际 2 个", result.stderr)

    def test_duplicate_source_across_files_is_rejected(self):
        self.page()
        shutil.copyfile(self.source_file, self.root / "sources" / "duplicate.md")
        with self.assertRaisesRegex(ValueError, "重复来源 ID"):
            wiki.sources(self.root)

    def test_duplicate_json_key_is_rejected(self):
        path = self.page()
        content = path.read_text(encoding="utf-8")
        path.write_text(content.replace('"id": "algorithm-example"',
                                        '"id": "algorithm-hidden", "id": "algorithm-example"'), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "重复"):
            wiki.pages(self.root)

    def test_wrong_page_type_is_a_validation_error(self):
        for value in (["algorithm"], {"type": "algorithm"}, None):
            with self.subTest(page_type=value):
                self.page(type=value)
                self.assertTrue(any("类型" in error for error in self.errors()))

    def test_unknown_related_and_source_are_rejected(self):
        self.page(related=["algorithm-missing"], sources=["paper-missing"])
        errors = self.errors()
        self.assertTrue(any("related 未登记 algorithm-missing" in error for error in errors))
        self.assertTrue(any("sources 未登记 paper-missing" in error for error in errors))

    def test_reference_arrays_reject_wrong_container_or_elements(self):
        for field in ("tags", "aliases", "related", "sources"):
            for value in ("scalar", [7], [""], [None], [{"id": "wrong"}]):
                with self.subTest(field=field, value=value):
                    self.page(**{field: value})
                    self.assertTrue(any(f"{field} 必须" in error for error in self.errors()))

    def test_repeated_array_entry_is_rejected(self):
        for field in ("tags", "aliases", "related", "sources"):
            with self.subTest(field=field):
                self.page(**{field: ["same", "same"]})
                self.assertTrue(any(f"{field} 存在重复项" in error for error in self.errors()))

    def test_dates_require_calendar_date_and_hyphenated_format(self):
        for value in ("2026-02-30", "2026-13-01", "20261023", "2026-W43-5", None, 20261023):
            with self.subTest(updated=value):
                self.page(updated=value)
                self.assertTrue(any("日期非法" in error for error in self.errors()))

    def test_missing_local_link_is_rejected_but_external_and_code_are_ignored(self):
        self.page(body="# 示例\n[缺失](missing.md)\n[远端](https://example.com/no-file)\n"
                       "```markdown\n[示例](not-a-real-link.md)\n```\n")
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("失效本地链接 missing.md", errors[0])

    def test_missing_local_link_with_spaces_is_rejected(self):
        self.page(body="# 示例\n[缺失](<missing file.md>)\n")
        self.assertTrue(any("失效本地链接" in error for error in self.errors()))

    def test_existing_space_and_percent_encoded_links_are_valid(self):
        (self.root / "wiki" / "notes file.txt").write_text("内容", encoding="utf-8")
        self.page(body="# 示例\n[尖括号](<notes file.txt>)\n[编码](notes%20file.txt)\n")
        self.assertEqual(self.errors(), [])

    def test_missing_percent_encoded_link_is_rejected(self):
        self.page(body="# 示例\n[缺失](missing%20file.md)\n")
        self.assertTrue(any("失效本地链接" in error for error in self.errors()))

    def test_heading_fragment_valid_same_and_other_page(self):
        self.page("target", id="algorithm-target", body="# Target\n\n## Valid Heading\n正文")
        self.page(body="# 示例\n\n## Local Heading\n"
                       "[本页](#local-heading)\n[其他页](target.md#valid-heading)\n")
        self.assertEqual(self.errors(), [])

    def test_missing_heading_fragment_is_rejected(self):
        self.page("target", id="algorithm-target", body="# Target\n\n## Valid Heading\n正文")
        self.page(body="# 示例\n[本页](#missing-heading)\n[其他页](target.md#missing-heading)\n")
        errors = self.errors()
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(all("missing-heading" in error for error in errors))

    def test_index_check_detects_missing_and_stale_metadata(self):
        self.page()
        missing = self.cli("build_index.py", "--check")
        self.assertNotEqual(missing.returncode, 0)
        built = self.cli("build_index.py")
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertEqual(self.cli("build_index.py", "--check").returncode, 0)
        index = self.root / "index.generated.md"
        before = index.read_text(encoding="utf-8")
        self.page(title="更新标题")
        stale = self.cli("build_index.py", "--check")
        self.assertNotEqual(stale.returncode, 0)
        self.assertIn("过期", stale.stderr)
        self.assertEqual(index.read_text(encoding="utf-8"), before)
        self.assertEqual(self.cli("build_index.py").returncode, 0)
        self.assertEqual(self.cli("build_index.py", "--check").returncode, 0)


if __name__ == "__main__":
    unittest.main()

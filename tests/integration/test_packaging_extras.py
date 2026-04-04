import re
import unittest
from pathlib import Path


def _repo_root():
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find repository root with pyproject.toml")


ROOT = _repo_root()
PYPROJECT = ROOT / "pyproject.toml"


def _read_pyproject():
    return PYPROJECT.read_text(encoding="utf-8")


def _section(text, name):
    pattern = rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[|\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        return ""
    return match.group(1)


class TestPackagingExtras(unittest.TestCase):
    def test_packages_include_only_saltai(self):
        text = _read_pyproject()

        self.assertIn('{ include = "saltai" }', text)
        self.assertNotIn("saltai_ext", text)

    def test_optional_dependencies_are_optional(self):
        text = _read_pyproject()
        deps = _section(text, "tool.poetry.dependencies")

        self.assertRegex(deps, r"clearml\s*=\s*\{[^}]*optional\s*=\s*true[^}]*\}")
        self.assertRegex(deps, r"boto3\s*=\s*\{[^}]*optional\s*=\s*true[^}]*\}")

    def test_extras_are_correct(self):
        text = _read_pyproject()
        extras = _section(text, "tool.poetry.extras")

        self.assertIn('clearml = ["clearml"]', extras)
        self.assertIn('s3 = ["boto3"]', extras)
        self.assertIn('all = ["clearml", "boto3"]', extras)


if __name__ == "__main__":
    unittest.main()

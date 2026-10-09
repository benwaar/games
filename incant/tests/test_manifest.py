"""Tests for manifest parser and topological sort."""

import tempfile
from pathlib import Path

import pytest

from incant.manifest import Manifest, parse_manifest, topological_sort
from incant.sigil import Sigil, SigilParam, SigilSignature, SigilTest


def _make_sigil(name, target="wat", deps=None):
    return Sigil(
        name=name,
        description=f"test {name}",
        target=target,
        signature=SigilSignature(
            inputs=[SigilParam(name="a", type="i32")],
            output_type="i32",
        ),
        tests=[SigilTest(inputs={"a": 1}, expect=1)],
        export=name,
        dependencies=deps or [],
    )


def _write_sigil_yaml(dir_path: Path, name: str, target: str = "wat") -> Path:
    path = dir_path / f"{name}.sigil.yaml"
    path.write_text(f"""\
name: {name}
description: test {name}
target: {target}

signature:
  inputs:
    - name: a
      type: i32
  output:
    type: i32

export: {name}

tests:
  - inputs: {{ a: 1 }}
    expect: 1
""")
    return path


class TestParseManifest:
    def test_basic_manifest(self, tmp_path):
        s1 = _write_sigil_yaml(tmp_path, "add")
        s2 = _write_sigil_yaml(tmp_path, "sub")

        manifest_path = tmp_path / "test.manifest.yaml"
        manifest_path.write_text(f"""\
name: math
target: wat
sigils:
  - {s1}
  - {s2}
""")

        m = parse_manifest(manifest_path)
        assert m.name == "math"
        assert m.target == "wat"
        assert len(m.sigils) == 2
        assert m.sigils[0].name == "add"
        assert m.sigils[1].name == "sub"

    def test_relative_paths(self, tmp_path):
        _write_sigil_yaml(tmp_path, "foo")

        manifest_path = tmp_path / "test.manifest.yaml"
        manifest_path.write_text("""\
name: test
target: wat
sigils:
  - foo.sigil.yaml
""")

        m = parse_manifest(manifest_path)
        assert m.sigils[0].name == "foo"

    def test_target_mismatch_raises(self, tmp_path):
        _write_sigil_yaml(tmp_path, "add", target="z80")

        manifest_path = tmp_path / "test.manifest.yaml"
        manifest_path.write_text(f"""\
name: test
target: wat
sigils:
  - {tmp_path / "add.sigil.yaml"}
""")

        with pytest.raises(ValueError, match="target"):
            parse_manifest(manifest_path)

    def test_empty_sigils_raises(self, tmp_path):
        manifest_path = tmp_path / "test.manifest.yaml"
        manifest_path.write_text("""\
name: empty
target: wat
sigils: []
""")

        with pytest.raises(ValueError, match="no sigils"):
            parse_manifest(manifest_path)


class TestTopologicalSort:
    def test_no_deps(self):
        a = _make_sigil("a")
        b = _make_sigil("b")
        result = topological_sort([a, b])
        assert set(s.name for s in result) == {"a", "b"}

    def test_linear_deps(self):
        a = _make_sigil("a")
        b = _make_sigil("b", deps=["a"])
        c = _make_sigil("c", deps=["b"])
        result = topological_sort([c, b, a])
        names = [s.name for s in result]
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")

    def test_diamond_deps(self):
        a = _make_sigil("a")
        b = _make_sigil("b", deps=["a"])
        c = _make_sigil("c", deps=["a"])
        d = _make_sigil("d", deps=["b", "c"])
        result = topological_sort([d, c, b, a])
        names = [s.name for s in result]
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("d")
        assert names.index("c") < names.index("d")

    def test_circular_raises(self):
        a = _make_sigil("a", deps=["b"])
        b = _make_sigil("b", deps=["a"])
        with pytest.raises(ValueError, match="Circular"):
            topological_sort([a, b])

    def test_unknown_dep_raises(self):
        a = _make_sigil("a", deps=["missing"])
        with pytest.raises(ValueError, match="Unknown dependency"):
            topological_sort([a])

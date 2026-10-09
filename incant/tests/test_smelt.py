"""Tests for BDD spec parser and library catalogue."""

import textwrap
from pathlib import Path

import pytest

from incant.smelt import Scenario, Spec, format_catalogue, load_catalogue, parse_spec


@pytest.fixture
def spec_file(tmp_path):
    """Write a BDD spec file and return its path."""
    def _write(content: str) -> Path:
        p = tmp_path / "test.spec.md"
        p.write_text(textwrap.dedent(content))
        return p
    return _write


class TestParseSpec:
    def test_basic_spec(self, spec_file):
        p = spec_file("""\
            ---
            target: wat
            ---
            # Greet user

            ## Scenario: basic greeting
            Given the program starts
            When the user enters "Ben"
            Then output "hello Ben"
        """)
        spec = parse_spec(p)
        assert spec.title == "Greet user"
        assert spec.target == "wat"
        assert len(spec.scenarios) == 1
        s = spec.scenarios[0]
        assert s.name == "basic greeting"
        assert s.given == ["the program starts"]
        assert s.when == ['the user enters "Ben"']
        assert s.then == ['output "hello Ben"']

    def test_multiple_scenarios(self, spec_file):
        p = spec_file("""\
            ---
            target: wat
            ---
            # Calculator

            ## Scenario: add
            Given the program starts
            When the user enters "2 3"
            Then output "5"

            ## Scenario: negative
            Given the program starts
            When the user enters "-1 1"
            Then output "0"
        """)
        spec = parse_spec(p)
        assert len(spec.scenarios) == 2
        assert spec.scenarios[0].name == "add"
        assert spec.scenarios[1].name == "negative"

    def test_default_target(self, spec_file):
        p = spec_file("""\
            # No frontmatter

            ## Scenario: test
            Given something
            When action
            Then result
        """)
        spec = parse_spec(p)
        assert spec.target == "wat"

    def test_z80_target(self, spec_file):
        p = spec_file("""\
            ---
            target: z80
            ---
            # Z80 program

            ## Scenario: test
            Given something
            When action
            Then result
        """)
        spec = parse_spec(p)
        assert spec.target == "z80"

    def test_and_continues_previous_clause(self, spec_file):
        p = spec_file("""\
            ---
            target: wat
            ---
            # Multi-step

            ## Scenario: test
            Given the program starts
            And memory is cleared
            When the user enters "hello"
            And the user enters "world"
            Then output "hello"
            And output "world"
        """)
        spec = parse_spec(p)
        s = spec.scenarios[0]
        assert s.given == ["the program starts", "memory is cleared"]
        assert s.when == ['the user enters "hello"', 'the user enters "world"']
        assert s.then == ['output "hello"', 'output "world"']

    def test_no_title_raises(self, spec_file):
        p = spec_file("""\
            ## Scenario: orphan
            Given something
            When action
            Then result
        """)
        with pytest.raises(ValueError, match="No title"):
            parse_spec(p)

    def test_no_scenarios_raises(self, spec_file):
        p = spec_file("""\
            # Empty spec
        """)
        with pytest.raises(ValueError, match="No scenarios"):
            parse_spec(p)

    def test_output_prefix_stripped(self, spec_file):
        p = spec_file("""\
            ---
            target: wat
            ---
            # Test

            ## Scenario: check
            Given the program starts
            When the user enters "x"
            Then output "result"
        """)
        spec = parse_spec(p)
        assert spec.scenarios[0].then == ['output "result"']


class TestLoadCatalogue:
    def test_empty_dir(self, tmp_path):
        cat = load_catalogue(tmp_path / "nonexistent")
        assert cat == {}

    def test_loads_sigils_by_target(self, tmp_path):
        libs = tmp_path / "libs"
        wat_dir = libs / "wat"
        wat_dir.mkdir(parents=True)
        (wat_dir / "add.sigil.yaml").write_text(textwrap.dedent("""\
            name: add
            description: Add two integers
            target: wat
            signature:
              inputs:
                - name: a
                  type: i32
                - name: b
                  type: i32
              output:
                type: i32
            export: add
            tests:
              - inputs: { a: 1, b: 2 }
                expect: 3
        """))
        cat = load_catalogue(libs)
        assert "wat" in cat
        assert len(cat["wat"]) == 1
        assert cat["wat"][0].name == "add"


class TestFormatCatalogue:
    def test_format_empty(self):
        assert format_catalogue({}, "wat") == ""

    def test_format_with_sigils(self, tmp_path):
        libs = tmp_path / "libs"
        wat_dir = libs / "wat"
        wat_dir.mkdir(parents=True)
        (wat_dir / "add.sigil.yaml").write_text(textwrap.dedent("""\
            name: add
            description: Add two integers
            target: wat
            signature:
              inputs:
                - name: a
                  type: i32
                - name: b
                  type: i32
              output:
                type: i32
            export: add
            tests:
              - inputs: { a: 1, b: 2 }
                expect: 3
        """))
        cat = load_catalogue(libs)
        text = format_catalogue(cat, "wat")
        assert "add" in text
        assert "i32" in text
        assert "Add two integers" in text

"""Smelt — BDD spec parser, library catalogue, and LLM decomposition."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .gen import call_llm
from .sigil import Sigil, parse_sigil


@dataclass
class Scenario:
    name: str
    given: list[str] = field(default_factory=list)
    when: list[str] = field(default_factory=list)
    then: list[str] = field(default_factory=list)


@dataclass
class Spec:
    title: str
    target: str
    scenarios: list[Scenario]


def parse_spec(path: Path) -> Spec:
    """Parse a BDD .spec.md file into a Spec object."""
    text = path.read_text()

    # Extract YAML frontmatter
    target = "wat"
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if frontmatter_match:
        fm = yaml.safe_load(frontmatter_match.group(1))
        if fm and "target" in fm:
            target = fm["target"]
        text = text[frontmatter_match.end():]

    # Extract title from first # heading
    title = ""
    lines = text.strip().split("\n")
    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            title = line[2:].strip()
            break

    if not title:
        raise ValueError(f"No title (# heading) found in {path}")

    # Parse scenarios
    scenarios: list[Scenario] = []
    current: Scenario | None = None

    for line in lines:
        line = line.strip()

        scenario_match = re.match(r"^##\s+Scenario:\s*(.+)", line)
        if scenario_match:
            if current:
                scenarios.append(current)
            current = Scenario(name=scenario_match.group(1).strip())
            continue

        if current is None:
            continue

        if line.startswith("Given "):
            current.given.append(line[6:])
        elif line.startswith("When "):
            current.when.append(line[5:])
        elif line.startswith("Then "):
            current.then.append(line[5:])
        elif line.startswith("And "):
            # "And" continues the previous clause type
            clause = line[4:]
            if current.then:
                current.then.append(clause)
            elif current.when:
                current.when.append(clause)
            elif current.given:
                current.given.append(clause)

    if current:
        scenarios.append(current)

    if not scenarios:
        raise ValueError(f"No scenarios found in {path}")

    return Spec(title=title, target=target, scenarios=scenarios)


def load_catalogue(libs_dir: Path) -> dict[str, list[Sigil]]:
    """Load reusable sigils from the library directory.

    Returns {target: [sigils]} mapping.
    """
    catalogue: dict[str, list[Sigil]] = {}

    if not libs_dir.exists():
        return catalogue

    for target_dir in sorted(libs_dir.iterdir()):
        if not target_dir.is_dir():
            continue
        target_name = target_dir.name
        sigils = []
        for sigil_path in sorted(target_dir.glob("*.sigil.yaml")):
            sigils.append(parse_sigil(sigil_path))
        if sigils:
            catalogue[target_name] = sigils

    return catalogue


def format_catalogue(catalogue: dict[str, list[Sigil]], target: str) -> str:
    """Format library sigils as text context for the LLM."""
    sigils = catalogue.get(target, [])
    if not sigils:
        return ""

    lines = ["## Available library functions\n"]
    for s in sigils:
        params = ", ".join(f"{p.name}: {p.type}" for p in s.signature.inputs)
        lines.append(f"- **{s.name}**({params}) → {s.signature.output_type}")
        lines.append(f"  {s.description}")
        if s.dependencies:
            lines.append(f"  Dependencies: {', '.join(s.dependencies)}")
    return "\n".join(lines)


DECOMPOSE_SYSTEM_PROMPT = """\
You are a software architect that decomposes BDD specifications into sigil YAML files.

A sigil is a YAML spec describing one function. The pipeline will generate code from each sigil.

You output ONLY valid YAML — no prose, no markdown fences, no explanation.
Output a YAML document with two keys:
- "sigils": a list of sigil objects
- "manifest": a manifest object with name, target, and sigil names

Each sigil object has these fields:
- name: unique function name (snake_case)
- description: what the function does (detailed enough for an LLM to generate code)
- target: "wat" or "z80"
- wasi: true (for the main entry point that does I/O)
- signature: { inputs: [{name, type}], output: {type} }
- export: the export name (same as name, for WAT)
- dependencies: list of other sigil names this calls (optional)
- tests: list of {inputs: {stdin: "..."}, expect: "..."} for WASI programs

For WASI programs:
- The main function has wasi: true, reads stdin, writes stdout
- Test inputs use {stdin: "input string"} and expect is the expected stdout string
- The signature for WASI main: inputs [{name: stdin, type: i32}], output {type: i32}
- Helper functions (non-WASI) use normal signature and tests

Rules:
- Check the library catalogue — reuse existing functions as dependencies instead of regenerating
- Only generate sigils for functions that don't exist in the library
- Every sigil must have at least one test
- Derive test cases from the BDD When/Then pairs
"""


def _format_spec_for_llm(spec: Spec) -> str:
    """Format parsed BDD spec as text for the LLM."""
    lines = [f"# {spec.title}", f"Target: {spec.target}", ""]
    for s in spec.scenarios:
        lines.append(f"## Scenario: {s.name}")
        for g in s.given:
            lines.append(f"Given {g}")
        for w in s.when:
            lines.append(f"When {w}")
        for t in s.then:
            lines.append(f"Then {t}")
        lines.append("")
    return "\n".join(lines)


def decompose_spec(
    spec: Spec,
    catalogue: dict[str, list[Sigil]],
    max_retries: int = 3,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use LLM to decompose a BDD spec into sigils + manifest.

    Returns (sigil_dicts, manifest_dict) — raw dicts ready to write as YAML.
    """
    spec_text = _format_spec_for_llm(spec)
    catalogue_text = format_catalogue(catalogue, spec.target)

    prompt = f"""{spec_text}

{catalogue_text}

Decompose this spec into sigil YAML files. Output ONLY a YAML document with "sigils" and "manifest" keys.
The manifest should list all sigil names (both new and from the library) needed to build this program.
"""

    last_error = None
    for attempt in range(1, max_retries + 1):
        if verbose:
            print(f"  Decompose attempt {attempt}/{max_retries}...")

        if last_error:
            retry_prompt = (
                f"{prompt}\n\n"
                f"Previous attempt had this error:\n{last_error}\n\n"
                f"Fix the error and output ONLY the corrected YAML."
            )
        else:
            retry_prompt = prompt

        raw_output = call_llm(DECOMPOSE_SYSTEM_PROMPT, retry_prompt)

        # Strip markdown fences if present
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            result = yaml.safe_load(cleaned)
        except yaml.YAMLError as e:
            last_error = f"YAML parse error: {e}"
            if verbose:
                print(f"  YAML parse failed: {e}")
            continue

        if not isinstance(result, dict):
            last_error = "Output is not a YAML mapping"
            continue

        if "sigils" not in result:
            last_error = "Missing 'sigils' key in output"
            continue

        if "manifest" not in result:
            last_error = "Missing 'manifest' key in output"
            continue

        sigils = result["sigils"]
        manifest = result["manifest"]

        # Validate sigils have required fields
        valid = True
        for i, s in enumerate(sigils):
            for field in ("name", "description", "target", "signature", "tests"):
                if field not in s:
                    last_error = f"Sigil {i} missing required field '{field}'"
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            if verbose:
                print(f"  Validation failed: {last_error}")
            continue

        if verbose:
            print(f"  Decomposed into {len(sigils)} sigils")

        return sigils, manifest

    raise ValueError(
        f"Decomposition failed after {max_retries} attempts. Last error: {last_error}"
    )


def write_sigils_and_manifest(
    sigil_dicts: list[dict[str, Any]],
    manifest_dict: dict[str, Any],
    output_dir: Path,
    libs_dir: Path,
) -> Path:
    """Write sigil YAML files and manifest to disk. Returns manifest path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sigil_paths: list[str] = []

    for s in sigil_dicts:
        name = s["name"]
        path = output_dir / f"{name}.sigil.yaml"
        with open(path, "w") as f:
            yaml.dump(s, f, default_flow_style=False, sort_keys=False)
        sigil_paths.append(str(path))

    # Add library sigils referenced in manifest
    manifest_sigil_names = set()
    if "sigils" in manifest_dict:
        manifest_sigil_names = set(manifest_dict["sigils"])

    generated_names = {s["name"] for s in sigil_dicts}
    lib_names = manifest_sigil_names - generated_names

    if lib_names and libs_dir.exists():
        target = manifest_dict.get("target", "wat")
        target_dir = libs_dir / target
        if target_dir.exists():
            for sigil_path in target_dir.glob("*.sigil.yaml"):
                sigil = parse_sigil(sigil_path)
                if sigil.name in lib_names:
                    sigil_paths.append(str(sigil_path))

    # Write manifest
    manifest_yaml = {
        "name": manifest_dict.get("name", "program"),
        "target": manifest_dict.get("target", "wat"),
        "sigils": sigil_paths,
    }
    manifest_path = output_dir / f"{manifest_yaml['name']}.manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest_yaml, f, default_flow_style=False, sort_keys=False)

    return manifest_path

"""Sigil parser — load YAML spec files."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SigilParam:
    name: str
    type: str
    register: str | None = None


@dataclass
class SigilSignature:
    inputs: list[SigilParam]
    output_type: str
    output_register: str | None = None


@dataclass
class SigilTest:
    inputs: dict[str, Any]
    expect: Any


@dataclass
class Sigil:
    name: str
    description: str
    target: str
    signature: SigilSignature
    tests: list[SigilTest]
    export: str | None = None
    memory: int | None = None
    dependencies: list[str] = field(default_factory=list)


def parse_sigil(path: Path) -> Sigil:
    with open(path) as f:
        raw = yaml.safe_load(f)

    sig_raw = raw["signature"]
    inputs = [
        SigilParam(
            name=p["name"],
            type=p["type"],
            register=p.get("register"),
        )
        for p in sig_raw["inputs"]
    ]

    output = sig_raw["output"]
    if isinstance(output, dict):
        output_type = output["type"]
        output_register = output.get("register")
    else:
        output_type = output
        output_register = None

    signature = SigilSignature(
        inputs=inputs,
        output_type=output_type,
        output_register=output_register,
    )

    tests = [
        SigilTest(inputs=t["inputs"], expect=t["expect"])
        for t in raw.get("tests", [])
    ]

    return Sigil(
        name=raw["name"],
        description=raw["description"],
        target=raw["target"],
        signature=signature,
        tests=tests,
        export=raw.get("export"),
        memory=raw.get("memory"),
        dependencies=raw.get("dependencies", []),
    )

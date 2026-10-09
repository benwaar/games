"""Manifest parser — load multi-sigil build specs."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .sigil import Sigil, parse_sigil


@dataclass
class Manifest:
    name: str
    target: str
    sigils: list[Sigil]
    sigil_paths: list[Path]


def parse_manifest(path: Path) -> Manifest:
    with open(path) as f:
        raw = yaml.safe_load(f)

    name = raw["name"]
    target = raw["target"]
    sigil_paths_raw = raw["sigils"]

    base_dir = path.parent
    sigils = []
    resolved_paths = []

    for sp in sigil_paths_raw:
        sigil_path = Path(sp)
        if not sigil_path.is_absolute():
            sigil_path = base_dir / sigil_path

        sigil = parse_sigil(sigil_path)

        if sigil.target != target:
            raise ValueError(
                f"Sigil '{sigil.name}' has target '{sigil.target}', "
                f"but manifest target is '{target}'"
            )

        sigils.append(sigil)
        resolved_paths.append(sigil_path)

    if not sigils:
        raise ValueError(f"Manifest '{name}' has no sigils")

    return Manifest(
        name=name,
        target=target,
        sigils=sigils,
        sigil_paths=resolved_paths,
    )


def topological_sort(sigils: list[Sigil]) -> list[Sigil]:
    """Sort sigils so dependencies come before dependents."""
    by_name = {s.name: s for s in sigils}
    visited: set[str] = set()
    result: list[Sigil] = []
    in_progress: set[str] = set()

    def visit(name: str):
        if name in visited:
            return
        if name in in_progress:
            raise ValueError(f"Circular dependency involving '{name}'")

        in_progress.add(name)
        sigil = by_name.get(name)
        if sigil is None:
            raise ValueError(f"Unknown dependency: '{name}'")

        for dep in sigil.dependencies:
            visit(dep)

        in_progress.remove(name)
        visited.add(name)
        result.append(sigil)

    for s in sigils:
        visit(s.name)

    return result

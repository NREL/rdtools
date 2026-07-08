#!/usr/bin/env python
"""Verify that dependency declarations in ``pyproject.toml`` are self-consistent.

Checks that the lower bounds in ``[project.dependencies]`` match the pinned
versions in ``[tool.pixi.feature.min.pypi-dependencies]`` (which is consumed by
the ``dev-min`` pixi environment used to test the minimum supported versions).

Exits 0 on success, 1 with a diff and fix instructions on failure.

Usage
-----
    python scripts/check_dependencies.py

Or via the pixi task::

    pixi run -e dev check-deps
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from packaging.requirements import Requirement
from packaging.version import Version


PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def main() -> int:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    project_deps: list[str] = data["project"]["dependencies"]
    min_pins: dict[str, str] = (
        data["tool"]["pixi"]["feature"]["min"]["pypi-dependencies"]
    )

    # canonical name -> (lower-bound version string, Requirement) from
    # [project.dependencies]; skip self-references like "rdtools[extras]".
    bounds: dict[str, tuple[str, Requirement]] = {}
    for dep in project_deps:
        if dep.split()[0].lower().startswith("rdtools"):
            continue
        req = Requirement(dep)
        for spec in req.specifier:
            if spec.operator == ">=":
                bounds[req.name.lower()] = (spec.version, req)
                break

    # Normalise pinned versions from [tool.pixi.feature.min.pypi-dependencies]
    # by stripping the leading "==".
    pinned: dict[str, str] = {
        name.lower(): version.lstrip("=").strip()
        for name, version in min_pins.items()
    }

    problems: list[str] = []
    for name, (bound, req) in bounds.items():
        pin = pinned.get(name)
        if pin is None:
            problems.append(
                f"  - {name}: no pin in [tool.pixi.feature.min.pypi-dependencies]"
                f" (project.dependencies lower bound {bound})"
            )
            continue
        pin_v = Version(pin)
        bound_v = Version(bound)
        if pin_v not in req.specifier:
            problems.append(
                f"  - {name}: min-feature pin '{pin}' does not satisfy "
                f"project.dependencies constraint '{req.specifier}'"
            )
        elif pin_v != bound_v:
            problems.append(
                f"  - {name}: min-feature pin is '{pin}' but "
                f"project.dependencies lower bound is '{bound}'"
            )

    for name in sorted(set(pinned) - set(bounds)):
        problems.append(
            f"  - {name}: pinned in [tool.pixi.feature.min.pypi-dependencies]"
            f" but not present in [project.dependencies]"
        )

    if not problems:
        print(
            "OK: [tool.pixi.feature.min.pypi-dependencies] matches the lower "
            "bounds in [project.dependencies]."
        )
        return 0

    print(
        "ERROR: [tool.pixi.feature.min.pypi-dependencies] is out of sync with "
        "[project.dependencies]:\n"
    )
    for line in problems:
        print(line)
    print(
        "\n"
        "To fix, edit pyproject.toml so each package pinned under\n"
        "[tool.pixi.feature.min.pypi-dependencies] matches its lower bound in\n"
        "[project.dependencies], then regenerate the lockfile:\n"
        "\n"
        "    pixi install\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

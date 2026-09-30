"""Generate a dependency-free ``.py`` loader for an exported SSM.

A downstream consumer (e.g. the 3D-ultrasound reconstruction project) should not have to
install this package to *use* a model: :func:`load_model` and the fit helpers need only numpy
and ``scipy.spatial.cKDTree``. But :mod:`shape_modeling.analysis.export` imports from the rest
of the package, so it cannot simply be copied out.

This script assembles the loader by lifting the required top-level definitions **from the real
sources** (via ``ast``, by name) rather than by hand, so the shipped file cannot silently drift
from the library it mirrors. ``scripts/make_ssm_bundle.py`` runs it, then verifies the result
reproduces the package's own fit on the actual model before shipping it.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "shape_modeling" / "analysis"

# What to lift, in emission order. export_model() is deliberately left out: it needs ShapeModel
# (i.e. the build stack), and a consumer reads models, never writes them.
FROM_SURFACE = ["Surface", "_similarity_fit"]
FROM_EXPORT = [
    "FORMAT_VERSION", "Pose", "FitResult", "SurfaceFitResult", "ExportedModel",
    "load_model", "surface_distance", "_SIGN_FLIPS", "_principal_axes", "_init_surface_poses",
]

HEADER = '''"""Standalone loader for a shape model exported by the ShapeModeling package.

GENERATED FILE -- do not edit. Produced by ``scripts/make_standalone_loader.py``, which lifts
these definitions verbatim from ``shape_modeling/analysis/{{export,surface}}.py``. Regenerate it
rather than patching it here.

Requires only **numpy** and **scipy** (``scipy.spatial.cKDTree``). Usage::

    from {module} import load_model
    m = load_model("model.npz")
    fit = m.fit_surface(cloud_mm, n_modes=15)

{doc}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["ExportedModel", "FitResult", "Pose", "Surface", "SurfaceFitResult",
           "load_model", "surface_distance"]

'''


def _defs(path: Path, names: list[str]) -> dict[str, str]:
    """Source text of the named top-level definitions (class/def/assignment) in ``path``."""
    source = path.read_text()
    tree = ast.parse(source)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            target = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            target = t.id if isinstance(t, ast.Name) else None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        else:
            continue
        if target in names:
            segment = ast.get_source_segment(source, node, padded=True)
            assert segment is not None, f"no source for {target}"
            # Decorators are not part of the node segment in every Python version; re-attach.
            decorators = getattr(node, "decorator_list", [])
            if decorators:
                deco = "\n".join(
                    "@" + ast.get_source_segment(source, d) for d in decorators  # type: ignore[operator]
                )
                segment = f"{deco}\n{segment}"
            found[target] = segment
    missing = [n for n in names if n not in found]
    assert not missing, f"{path.name}: could not find {missing}"
    return found


def build(module_name: str = "thyroid_ssm") -> str:
    surface = _defs(SRC / "surface.py", FROM_SURFACE)
    export = _defs(SRC / "export.py", FROM_EXPORT)
    doc = ast.get_docstring(ast.parse((SRC / "export.py").read_text())) or ""
    # Keep only the parameterisation/pose explanation; drop the package-oriented first paragraph.
    doc = "\n\n".join(p for p in doc.split("\n\n") if p.startswith(("Parameterisation", "Pose at")))

    parts = [HEADER.format(module=module_name, doc=doc)]
    parts += [surface[n] for n in FROM_SURFACE]
    parts += [export[n] for n in FROM_EXPORT]
    return "\n\n".join(p.strip("\n") for p in parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path, help="path of the .py file to write")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(args.out.stem))
    print(f"wrote {args.out} ({len(args.out.read_text().splitlines())} lines)")


if __name__ == "__main__":
    main()

"""Reader/writer and de-normalisation helpers for MGA ORACLE polytope files.

This module is the single owner of the polytope npz schema written by the MGA
plugin and read by the thesis-side consumers (polytope sampling and the
interpretation suite). It is deliberately dependency-light (numpy + stdlib
only) and side-effect-free: importing it does NOT import the MGA plugin and
therefore registers no event handlers.

Schema (all plain arrays, no pickled objects):
    A               (n_rows, n_explore)  outer approximation, normalised coords
    b               (n_rows,)
    X               (n_points, n_explore) feasible points; X[0] is z* normalised
    name_list       (n_explore,) str     design axes first, cost axis last
    u_star          (n_z,)               per-design-axis fmax maxima (physical)
    c_star          ()                   baseline net_present_cost
    epsilon         ()                   near-optimality slack
    cost_axis       () str               "" when the run had no cost coordinate
    z_star          (n_z,)               raw physical baseline design vector
    units           (n_explore,) str     physical unit strings, "" if unknown
    tolerance       ()                   ORACLE convergence tolerance
    converged       () bool
    final_max_min_distance ()            NaN if the run raised before any result
    axis_meta_json  () str               json: per-axis kind/members/capacity_type/unit

Normalisation convention (fixed): design axis i is z_i / u_star[i]; the cost
axis (when present) is (C - c_star) / (epsilon * c_star). The augmented scale
and offset (u_tilde = [u_star..., eps*c_star], offset = [0..., c_star]) are an
exact repackaging of the stored scalars and are therefore not stored.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Axis-kind strings used in axis_meta_json (and by the MGA plugin's Axis type).
TECH_CAPACITY = "tech_capacity"
CARRIER_IMPORT = "carrier_import"

NPZ_KEYS = (
    "A", "b", "X", "name_list", "u_star", "c_star", "epsilon", "cost_axis",
    "z_star", "units", "tolerance", "converged", "final_max_min_distance",
    "axis_meta_json",
)


@dataclass(frozen=True)
class Polytope:
    """A loaded (or to-be-written) ORACLE polytope with its de-norm constants."""

    A: np.ndarray
    b: np.ndarray
    X: np.ndarray
    names: list[str]
    u_star: np.ndarray
    c_star: float
    epsilon: float
    cost_axis: str            # "" when there is no cost coordinate (never None)
    z_star: np.ndarray
    units: list[str]
    tolerance: float
    converged: bool
    final_max_min_distance: float
    meta: dict = field(repr=False)  # full parsed axis_meta_json, round-trips verbatim

    @property
    def axes(self) -> list[dict]:
        """Per-design-axis metadata dicts (name, kind, members, capacity_type, unit)."""
        return self.meta["axes"]

    @property
    def n_z(self) -> int:
        """Number of design axes (excludes the cost coordinate)."""
        return len(self.u_star)

    @property
    def design_names(self) -> list[str]:
        return [n for n in self.names
                if not (self.cost_axis and n == self.cost_axis)]

    def to_phys(self, Z: np.ndarray) -> np.ndarray:
        """Map normalised coordinates (..., n_explore) to physical units."""
        return norm_to_phys(Z, self.names, self.u_star, self.c_star,
                            self.epsilon, self.cost_axis)

    def to_norm(self, Z: np.ndarray) -> np.ndarray:
        """Inverse of to_phys."""
        return phys_to_norm(Z, self.names, self.u_star, self.c_star,
                            self.epsilon, self.cost_axis)


def save_polytope(path, poly: Polytope) -> None:
    """Write `poly` to `path` (npz, exact NPZ_KEYS schema)."""
    np.savez(
        Path(path),
        A=poly.A, b=poly.b, X=poly.X,
        name_list=np.array(poly.names),
        u_star=poly.u_star,
        c_star=float(poly.c_star),
        epsilon=float(poly.epsilon),
        cost_axis=np.array(poly.cost_axis or ""),
        z_star=poly.z_star,
        units=np.array(poly.units),
        tolerance=float(poly.tolerance),
        converged=bool(poly.converged),
        final_max_min_distance=float(poly.final_max_min_distance),
        axis_meta_json=np.array(json.dumps(poly.meta)),
    )


def load_polytope(path) -> Polytope:
    """Load a polytope npz written by save_polytope.

    Raises KeyError for files missing schema keys (re-run ORACLE with the
    current MGA plugin — there is no support for older formats) and ValueError
    for internally inconsistent shapes.
    """
    path = Path(path)
    d = np.load(path)  # schema has no object arrays; allow_pickle stays False
    missing = [k for k in NPZ_KEYS if k not in d.files]
    if missing:
        raise KeyError(
            f"{path} is missing {missing}; re-run ORACLE with the current MGA "
            f"plugin (the file does not match the polytope schema)."
        )
    A = np.asarray(d["A"], dtype=float)
    b = np.asarray(d["b"], dtype=float).ravel()
    X = np.asarray(d["X"], dtype=float)
    names = [str(n) for n in d["name_list"]]
    u_star = np.asarray(d["u_star"], dtype=float).ravel()
    z_star = np.asarray(d["z_star"], dtype=float).ravel()
    units = [str(u) for u in d["units"]]
    cost_axis = str(d["cost_axis"])
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"{path}: b has {b.shape[0]} rows, A has {A.shape[0]}")
    if not (X.shape[1] == A.shape[1] == len(names) == len(units)):
        raise ValueError(
            f"{path}: inconsistent axis counts (A: {A.shape[1]}, X: {X.shape[1]}, "
            f"names: {len(names)}, units: {len(units)})"
        )
    if len(u_star) != len(z_star) or len(u_star) != len(names) - bool(cost_axis):
        raise ValueError(
            f"{path}: u_star/z_star length {len(u_star)}/{len(z_star)} does not "
            f"match {len(names)} axes with cost_axis={cost_axis!r}"
        )
    return Polytope(
        A=A, b=b, X=X, names=names, u_star=u_star,
        c_star=float(d["c_star"]), epsilon=float(d["epsilon"]),
        cost_axis=cost_axis, z_star=z_star, units=units,
        tolerance=float(d["tolerance"]), converged=bool(d["converged"]),
        final_max_min_distance=float(d["final_max_min_distance"]),
        meta=json.loads(str(d["axis_meta_json"])),
    )


# ---------------------------------------------------------------------------
# Normalised <-> physical conversion (standalone, array-shape (..., len(names)))
# ---------------------------------------------------------------------------

def norm_to_phys(Z, names, u_star, c_star, epsilon, cost_axis="") -> np.ndarray:
    """Map normalised coordinates to physical units.

    Works for any leading shape (..., len(names)). The column whose name equals
    a non-empty `cost_axis` maps c_star*(1 + eps*z); every other column consumes
    the next u_star entry (z * u). For a design-only subset pass the design
    names and cost_axis="".
    """
    Z = np.asarray(Z, dtype=float)
    if Z.shape[-1] != len(names):
        raise ValueError(
            f"last dimension {Z.shape[-1]} does not match {len(names)} axes"
        )
    out = Z.copy()
    design = 0
    for j, nm in enumerate(names):
        if cost_axis and nm == cost_axis:
            out[..., j] = c_star * (1.0 + epsilon * Z[..., j])
        else:
            out[..., j] = Z[..., j] * u_star[design]
            design += 1
    return out


def phys_to_norm(Z, names, u_star, c_star, epsilon, cost_axis="") -> np.ndarray:
    """Inverse of norm_to_phys (same column semantics)."""
    Z = np.asarray(Z, dtype=float)
    if Z.shape[-1] != len(names):
        raise ValueError(
            f"last dimension {Z.shape[-1]} does not match {len(names)} axes"
        )
    out = Z.copy()
    design = 0
    for j, nm in enumerate(names):
        if cost_axis and nm == cost_axis:
            out[..., j] = (Z[..., j] / c_star - 1.0) / epsilon
        else:
            out[..., j] = Z[..., j] / u_star[design]
            design += 1
    return out


def axis_norm_to_phys(value, axis_name, names, u_star, c_star, epsilon,
                      cost_axis="") -> float:
    """Convert a single normalised value on the named axis to physical units."""
    if cost_axis and axis_name == cost_axis:
        return float(c_star * (1.0 + epsilon * value))
    design_names = [n for n in names if not (cost_axis and n == cost_axis)]
    j = design_names.index(axis_name)  # ValueError for unknown names
    return float(value * u_star[j])

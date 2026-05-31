"""
MGA (Modeling to Generate Alternatives) plugin for ZEN-garden.

Registers an `after_solve` event handler that, after the baseline optimization
has been solved, adds a near-optimality cost constraint and re-solves the
problem under one of three exploration modes selected via config:

    * "weights" (existing): user-provided list of per-iteration weight dicts;
      each iteration solves min sum_i w_i * cap_add_i.
    * "oracle": ORACLE algorithm from Turan, Moret, Bardow (2026).
      Iteratively refines inner/outer polytope approximations of Z_eps via
      L-infinity projections of trial points. Delegates the loop to
      pyoNearOpt.exploration_methods.ORACLE.oracle.

    ("random_directions" was retired when the legacy capacity-based
    outer-approximation bounds were removed — it never ran the fmax LPs and so
    has no bound source. The branch now raises NotImplementedError.)

For ORACLE, the initial outer polytope is the per-axis fmax box: one upper
bound `z_i <= U_i*` per axis (from the fmax LPs, which also supply the
normalisation denominators), plus per-axis non-negativity `z_i >= 0`, plus —
when `include_cost=true` — two cost-coordinate rows (`C <= (1+eps)*C*` and
`C >= C*`). Coordinates are ALWAYS normalised (z_i / U_i*), so the box is the
unit cube in normalised space. The near-optimal cost budget c^T x <= (1+eps)C*
is enforced inside the model (added by setup()), not as a polytope row. Each
axis is either a tech-capacity axis (sum of capacity_addition over member
techs; singleton or lumped via `include_techs`) or a carrier-import axis
(duration-weighted annual flow_import over member carriers, via
`include_carrier_imports`). See `MGA.build_initial_outer_approximation`.

Each iteration is written to disk as a sibling sub-solution next to the
baseline, using Postprocess with a modified model_name.

Configuration (via the "plugins.mga" block in config.json):
    epsilon (float): near-optimality slack. Default 0.1.
    mode (str): "weights" | "oracle". Default "weights".
    exclude_techs (list[str]): technologies excluded from the tech-capacity
        axes. oracle mode only. Mutually exclusive with include_techs.
    include_techs (list): positive list of tech-capacity axes. Alternative to
        exclude_techs (mutually exclusive). Each entry is either a bare string
        (singleton tech axis) or a single-key dict
        {group_name: [member_tech_1, ...]} (a lumped axis: one z-axis equal to
        the sum of capacity_addition over all members).
        Example:
            "include_techs": [
                "nuclear", "photovoltaics",
                {"ccs_lump": ["BF_BOF_CCS", "SMR_CCS",
                              "natural_gas_turbine_CCS"]}
            ]
    include_carrier_imports (list): positive list of carrier-import axes,
        additive with the tech axes. Each entry is a bare carrier string
        (singleton axis = that carrier's duration-weighted annual import) or a
        single-key dict {name: [carrier_1, ...]} (a lumped carrier axis). The
        axis value is sum_{m,n,t} tau_t * flow_import[m, n, t] (GWh/yr).
        Example: "include_carrier_imports": ["biomass"].
    iterations (list[dict]): used only in weights mode. One entry per MGA
        iteration with a "weights": {tech_name: float} dict.
    oracle (dict): used only in oracle mode. Keys:
        max_iterations (int), tolerance (float, REQUIRED — no default),
        Md_override (float), t_max_override (float|None).
        include_cost (bool): default False. When True the exploration vector
            is augmented with a total-cost coordinate (the model's
            net_present_cost), normalised to (C - C*) / (eps * C*).
        (normalization is ALWAYS on for ORACLE — every axis is normalised by
        its fmax maximum U_i*. A legacy "normalization" key is ignored with a
        deprecation note.)
        Solver is hardcoded to Gurobi.
"""

import hashlib
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.postprocess.postprocess import Postprocess


# Module-level config dict. The plugin loader merges user values from
# config.json's "plugins.mga" block via dict.update (SHALLOW). Therefore
# defaults for nested dicts like "random_directions" / "oracle" must be
# applied at access time via .get(...), NOT relied upon to survive the
# loader merge.
config = {
    "epsilon": 0.1,
    "mode": "weights",
    "exclude_techs": [],
    "include_techs": [],
    "include_carrier_imports": [],
    "iterations": [],
    "oracle": {},
}


class MGA:
    """MGA core logic.

    Holds the linopy model and helper methods shared by the three exploration
    modes. The split between `setup()` and the per-iteration methods exists
    because `linopy.Model.add_constraints` is not idempotent: the
    near-optimality constraint is added exactly once in `setup()`, while the
    objective is swapped per iteration via `add_objective(..., overwrite=True)`.

    For `oracle` mode, an additional projection-model variable (delta), a
    scalar t, and three constraints are added by `setup_projection_model()`.
    """

    def __init__(self, optimization_setup, epsilon, postprocess_ctx, exclude_techs=None,
                 include_techs=None, include_carrier_imports=None, include_cost=False):
        """
        Args:
            optimization_setup: The OptimizationSetup instance, expected to
                already hold the solved baseline (so model.objective.value
                equals C*).
            epsilon: Near-optimality slack, e.g. 0.1 for a 10% cost budget.
            postprocess_ctx: Dict with keys "scenarios", "subfolder",
                "model_name", "scenario_name", "param_map". Forwarded to
                Postprocess for each iteration's output.
            exclude_techs: Optional list of technology names to drop from the
                tech-capacity exploration axes. Mutually exclusive with
                include_techs.
            include_techs: Optional positive list of tech-capacity axes. Each
                entry is either a bare string (singleton axis = that tech) or a
                single-key dict ``{name: [members]}`` (lumped axis = sum of
                capacity_addition over members). Mutually exclusive with
                exclude_techs.
            include_carrier_imports: Optional list of carrier-import axes. Each
                entry is a bare carrier string (singleton axis = that carrier's
                duration-weighted annual import) or a single-key dict
                ``{name: [carriers]}`` (lumped axis = sum over carriers).
                Additive with the tech axes (does not interact with
                include/exclude_techs).
            include_cost: if True, augment z with a total-cost coordinate.
                oracle mode only.

        ORACLE mode always normalises z by the per-axis fmax maxima U_i*; there
        is no raw-coordinate mode.
        """
        self.optimization_setup = optimization_setup
        self.model = optimization_setup.model
        self.cap_add = self.model.variables["capacity_addition"]
        if epsilon <= 0:
            raise ValueError(f"MGA epsilon must be positive, got {epsilon!r}")
        self.epsilon = epsilon
        # C* captured here, before any MGA modifications to model.objective.
        self.c_star = self.model.objective.value
        self.postprocess_ctx = postprocess_ctx

        self.include_cost = include_cost
        # fmax state, populated by compute_fmax_normalization():
        #   u_star   - per-axis U_g* on z_names order
        #   _u_tilde - augmented scale (u_star, [eps*C*]) on explore order
        #   _offset  - augmented offset (0..., [C*]) on explore order
        # ORACLE always normalises; these must be set before any coordinate
        # transform (see _to_explore_coords, which raises if they are None).
        self.u_star = None
        self._u_tilde = None
        self._offset = None

        # --- Build the canonical list of exploration AXES ---
        # Each axis is a (name, members) tuple carrying a KIND in self.axis_kind:
        #   "tech_capacity"  -> members are technologies; the axis value is the
        #                       sum of capacity_addition over members.
        #   "carrier_import" -> members are carriers; the axis value is the
        #                       duration-weighted annual flow_import over members.
        # z_groups is the SINGLE source of truth for the ordering of z, mu,
        # name_list everywhere downstream. Tech axes come first (in the order
        # implied by include_techs/exclude_techs), then carrier axes (user order).
        all_techs = list(self.cap_add.coords["set_technologies"].values)
        all_techs_set = set(all_techs)
        # Carriers (for carrier-import axes + the group-name reserved set).
        if "flow_import" in self.model.variables:
            all_carriers = list(
                self.model.variables["flow_import"].coords["set_carriers"].values
            )
        else:
            all_carriers = []
        all_carriers_set = set(all_carriers)

        include_techs_list = list(include_techs or [])
        exclude_techs_list = list(exclude_techs or [])
        include_carrier_list = list(include_carrier_imports or [])

        if include_techs_list and exclude_techs_list:
            raise ValueError(
                "MGA: include_techs and exclude_techs are mutually exclusive "
                "(both non-empty)."
            )

        if include_techs_list:
            # include_techs path. Ordering follows USER order (not model
            # order) — the asymmetry is intentional; no downstream consumer
            # keys off model order.
            z_groups: list[tuple[str, list[str]]] = []
            unknown_techs: list[str] = []
            seen_techs: dict[str, str] = {}  # tech -> entry that first claimed it
            seen_names: set[str] = set()
            for idx, entry in enumerate(include_techs_list):
                if isinstance(entry, str):
                    if not entry:
                        raise ValueError(
                            f"MGA include_techs[{idx}]: empty string is not "
                            f"a valid tech name."
                        )
                    name = entry
                    members = [entry]
                elif isinstance(entry, dict):
                    if len(entry) != 1:
                        raise ValueError(
                            f"MGA include_techs[{idx}]: group dict must have "
                            f"exactly one key, got {len(entry)}: "
                            f"{sorted(entry.keys())}."
                        )
                    (name, members) = next(iter(entry.items()))
                    if not isinstance(name, str) or not name:
                        raise ValueError(
                            f"MGA include_techs[{idx}]: group name must be a "
                            f"non-empty string, got {name!r}."
                        )
                    if name in all_techs_set:
                        raise ValueError(
                            f"MGA include_techs[{idx}]: group name {name!r} "
                            f"collides with a technology in set_technologies. "
                            f"Rename the group."
                        )
                    if not isinstance(members, list) or not members:
                        raise ValueError(
                            f"MGA include_techs[{idx}] ({name!r}): group "
                            f"members must be a non-empty list of tech names, "
                            f"got {members!r}."
                        )
                    if not all(isinstance(m, str) and m for m in members):
                        raise ValueError(
                            f"MGA include_techs[{idx}] ({name!r}): every "
                            f"member must be a non-empty string, got "
                            f"{members!r}."
                        )
                else:
                    raise ValueError(
                        f"MGA include_techs[{idx}]: entry must be a string or "
                        f"a single-key dict, got {type(entry).__name__}: "
                        f"{entry!r}."
                    )
                if name in seen_names:
                    raise ValueError(
                        f"MGA include_techs[{idx}]: duplicate axis name "
                        f"{name!r}."
                    )
                seen_names.add(name)
                for m in members:
                    if m not in all_techs_set:
                        unknown_techs.append(m)
                    elif m in seen_techs:
                        raise ValueError(
                            f"MGA include_techs[{idx}] ({name!r}): tech "
                            f"{m!r} also appears in entry {seen_techs[m]!r}; "
                            f"each tech may appear at most once across all "
                            f"include_techs entries."
                        )
                    else:
                        seen_techs[m] = name
                z_groups.append((name, list(members)))
            if unknown_techs:
                raise KeyError(
                    f"MGA include_techs contains unknown technologies: "
                    f"{sorted(set(unknown_techs))}"
                )
        else:
            # Legacy path: singletons in MODEL order; byte-identical to today.
            excluded_set = set(exclude_techs_list)
            unknown = excluded_set - all_techs_set
            if unknown:
                raise KeyError(
                    f"exclude_techs contains unknown technologies: "
                    f"{sorted(unknown)}"
                )
            z_groups = [(t, [t]) for t in all_techs if t not in excluded_set]

        # Tech axes are now in z_groups. Record their kind and reserve their
        # names, then append carrier-import axes.
        axis_kind: dict[str, str] = {name: "tech_capacity" for name, _ in z_groups}
        seen_axis_names: set[str] = set(axis_kind)
        # Group/lump names must be a fresh label, not an existing tech OR
        # carrier name (the polytope name_list shares one namespace).
        reserved_names = all_techs_set | all_carriers_set

        # --- Carrier-import axes (additive; orthogonal to tech axes) ---
        unknown_carriers: list[str] = []
        seen_carriers: dict[str, str] = {}  # carrier -> axis that first claimed it
        for idx, entry in enumerate(include_carrier_list):
            if isinstance(entry, str):
                if not entry:
                    raise ValueError(
                        f"MGA include_carrier_imports[{idx}]: empty string is "
                        f"not a valid carrier name."
                    )
                name = entry
                members = [entry]
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(
                        f"MGA include_carrier_imports[{idx}]: group dict must "
                        f"have exactly one key, got {len(entry)}: "
                        f"{sorted(entry.keys())}."
                    )
                (name, members) = next(iter(entry.items()))
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"MGA include_carrier_imports[{idx}]: group name must "
                        f"be a non-empty string, got {name!r}."
                    )
                if name in reserved_names:
                    raise ValueError(
                        f"MGA include_carrier_imports[{idx}]: group name "
                        f"{name!r} collides with a technology or carrier name. "
                        f"Rename the group."
                    )
                if not isinstance(members, list) or not members:
                    raise ValueError(
                        f"MGA include_carrier_imports[{idx}] ({name!r}): group "
                        f"members must be a non-empty list of carrier names, "
                        f"got {members!r}."
                    )
                if not all(isinstance(m, str) and m for m in members):
                    raise ValueError(
                        f"MGA include_carrier_imports[{idx}] ({name!r}): every "
                        f"member must be a non-empty string, got {members!r}."
                    )
            else:
                raise ValueError(
                    f"MGA include_carrier_imports[{idx}]: entry must be a "
                    f"string or a single-key dict, got "
                    f"{type(entry).__name__}: {entry!r}."
                )
            if name in seen_axis_names:
                raise ValueError(
                    f"MGA include_carrier_imports[{idx}]: axis name {name!r} "
                    f"duplicates an existing axis (tech or carrier)."
                )
            seen_axis_names.add(name)
            for m in members:
                if m not in all_carriers_set:
                    unknown_carriers.append(m)
                elif m in seen_carriers:
                    raise ValueError(
                        f"MGA include_carrier_imports[{idx}] ({name!r}): "
                        f"carrier {m!r} also appears in axis "
                        f"{seen_carriers[m]!r}; each carrier may appear at "
                        f"most once across all carrier-import axes."
                    )
                else:
                    seen_carriers[m] = name
            z_groups.append((name, list(members)))
            axis_kind[name] = "carrier_import"
        if unknown_carriers:
            raise KeyError(
                f"MGA include_carrier_imports contains unknown carriers: "
                f"{sorted(set(unknown_carriers))}"
            )

        self.z_groups: list[tuple[str, list[str]]] = z_groups
        self.axis_kind: dict[str, str] = axis_kind
        self.z_names: list[str] = [name for name, _ in z_groups]
        self._members_by_name: dict[str, list[str]] = {n: m for n, m in z_groups}
        # Flat list of all tech-axis member techs (for the projection selector
        # matrix and the now-dead support_function). Carrier members excluded.
        self.z_techs: list[str] = [
            m for name, members in z_groups
            if axis_kind[name] == "tech_capacity" for m in members
        ]
        self.n_z: int = len(self.z_groups)
        self.has_carrier_axis: bool = any(
            k == "carrier_import" for k in axis_kind.values()
        )
        # True iff the per-axis dim cannot be "set_technologies": any tech lump,
        # any tech-axis name not in set_technologies, OR any carrier axis.
        # Drives the proj-eq dim-name choice in setup_projection_model
        # (set_technologies when False, else mga_z_axis).
        self._has_lumped_groups: bool = self.has_carrier_axis or any(
            len(members) > 1 or name not in all_techs_set
            for name, members in z_groups
            if axis_kind[name] == "tech_capacity"
        )

        # Model handles for carrier-import axes (looked up once).
        if self.has_carrier_axis:
            self.flow_import = self.model.variables["flow_import"]
            # Operational time-step duration tau_t (set_time_steps_operation).
            self._ts_duration = (
                self.optimization_setup.parameters.time_steps_operation_duration
            )
        else:
            self.flow_import = None
            self._ts_duration = None

        # Baseline design vector z*, captured NOW while the baseline solution is
        # still loaded (the fmax LPs overwrite it). Per-axis, kind-aware.
        self._z_star_design_raw = np.array(
            [self._axis_value(name) for name in self.z_names], dtype=float
        )

        # Sanity tripwire for carrier axes: the duration-weighted import must be
        # positive, finite, and DIFFERENT from the unweighted sum (the 10
        # aggregated op-steps have durations far from 1, so equality would mean
        # tau_t was silently dropped — the most likely mistake).
        for name in self.z_names:
            if self.axis_kind[name] != "carrier_import":
                continue
            members = self._members_by_name[name]
            weighted = self._axis_value(name)
            unweighted = float(
                self.flow_import.solution.sel(set_carriers=members).sum()
            )
            if not (np.isfinite(weighted) and weighted > 0):
                raise RuntimeError(
                    f"MGA carrier axis {name!r}: baseline duration-weighted "
                    f"import is {weighted!r}; expected a positive finite value."
                )
            if abs(weighted - unweighted) <= 1e-6 * max(1.0, abs(weighted)):
                raise RuntimeError(
                    f"MGA carrier axis {name!r}: duration-weighted import "
                    f"({weighted:.6g}) equals the unweighted sum "
                    f"({unweighted:.6g}). The time_steps_operation_duration "
                    f"weighting was not applied — this is a bug."
                )
            logging.info(
                f"MGA carrier axis {name!r}: baseline weighted import = "
                f"{weighted:.6g} (GWh/yr), unweighted sum = {unweighted:.6g} "
                f"(tripwire OK; members={members})."
            )

        # Iteration counter for labelling Postprocess folders.
        self._iter_count = 0

    def setup(self):
        """Add the near-optimality cost constraint. Call exactly once per run."""
        orig_cost_expr = self.optimization_setup.energy_system.rules.objective_total_cost(self.model)
        self.model.add_constraints(
            orig_cost_expr <= (1 + self.epsilon) * self.c_star,
            name="mga_near_optimality",
        )
        logging.info(
            f"MGA: added near-optimality constraint cost <= "
            f"{(1 + self.epsilon) * self.c_star} "
            f"(C* = {self.c_star}, epsilon = {self.epsilon})"
        )

    # ------------------------------------------------------------------
    # weights-mode methods (unchanged behaviour, kept as-is for backward
    # compatibility with existing configs and the v1 cb_full_mga run)
    # ------------------------------------------------------------------

    def run_iteration(self, weights: dict, iter_id: int):
        """Run one weights-mode MGA iteration."""
        w = self._build_w(weights)
        mga_obj = (w * self.cap_add).sum()
        self.model.add_objective(mga_obj, sense="min", overwrite=True)
        logging.info(f"MGA iter {iter_id}: objective replaced with weights = {weights}")

        self._solve_and_postprocess(f"mga_iter_{iter_id}")

    def _build_w(self, weights: dict) -> xr.DataArray:
        """Build a 1D DataArray over the FULL set_technologies coord from a dict.

        Used by weights mode. Unknown technology names raise KeyError.
        Missing technologies default to weight 0; (w * cap_add).sum()
        implicitly aggregates over (cap_type, location, year) via xarray
        broadcasting.
        """
        tech_coord = self.cap_add.coords["set_technologies"]
        w = xr.DataArray(
            np.zeros(tech_coord.size),
            dims=("set_technologies",),
            coords={"set_technologies": tech_coord},
        )
        known = set(tech_coord.values)
        for tech, val in weights.items():
            if tech not in known:
                raise KeyError(f"Unknown technology in MGA weights: {tech!r}")
            w.loc[tech] = float(val)
        return w

    # ------------------------------------------------------------------
    # Common helpers used by random_directions and oracle modes
    # ------------------------------------------------------------------

    @property
    def n_explore(self) -> int:
        """Dimension of the exploration vector handed to ORACLE (n_z, or n_z+1
        when include_cost augments z with a cost coordinate)."""
        return self.n_z + (1 if self.include_cost else 0)

    # Aggregation dims per axis kind (module-private constants inlined here).
    _TECH_AGG = ["set_capacity_types", "set_location", "set_time_steps_yearly"]
    _CARRIER_AGG = ["set_carriers", "set_nodes", "set_time_steps_operation"]

    def _axis_linexpr(self, name: str):
        """linopy LinearExpression for one axis (fmax LP objective / projection).

        Branches on axis kind (the ONLY place, besides _axis_value, that does):
          tech_capacity  -> sum of capacity_addition over the axis's member
                            techs and over (cap_type, loc, year).
          carrier_import -> duration-weighted sum of flow_import over the axis's
                            member carriers, nodes, and operational time steps:
                            sum_{m,n,t} tau_t * flow_import[m, n, t].
        """
        members = self._members_by_name[name]
        if self.axis_kind[name] == "tech_capacity":
            return (
                self.cap_add.sel(set_technologies=members)
                .sum(self._TECH_AGG + ["set_technologies"])
            )
        # carrier_import
        return (
            (self._ts_duration * self.flow_import.sel(set_carriers=members))
            .sum(self._CARRIER_AGG)
        )

    def _axis_value(self, name: str) -> float:
        """Scalar value of one axis on the currently loaded solution.

        Same expression as _axis_linexpr but evaluated on `.solution`; used for
        the frozen baseline z* and for _extract_z after each projection solve.
        """
        members = self._members_by_name[name]
        if self.axis_kind[name] == "tech_capacity":
            return float(
                self.cap_add.solution
                .sel(set_technologies=members)
                .sum(self._TECH_AGG + ["set_technologies"])
            )
        # carrier_import
        return float(
            (self._ts_duration * self.flow_import.solution.sel(set_carriers=members))
            .sum(self._CARRIER_AGG)
        )

    def _to_explore_coords(self, z_design_raw: np.ndarray, c_raw=None) -> np.ndarray:
        """Map a raw (design, cost) solution to ORACLE polytope coordinates.

        ORACLE always normalises: this applies the affine map
        (z - offset) / U_tilde, so design axes become z_i/U_i* and (when
        include_cost) the cost axis becomes (C - C*)/(eps*C*). The fmax LPs
        must have run first (u_star/_u_tilde/_offset populated).
        """
        if self._u_tilde is None:
            raise RuntimeError(
                "MGA: coordinate normalisation requested before the fmax LPs "
                "ran; compute_fmax_normalization() must be called first."
            )
        raw = np.asarray(z_design_raw, dtype=float)
        if self.include_cost:
            raw = np.append(raw, float(c_raw))
        return (raw - self._offset) / self._u_tilde

    def _extract_z(self) -> np.ndarray:
        """Read the exploration vector z from the most recent solve.

        Returns a 1D ndarray of length n_explore in the canonical z_groups
        order (cost appended last when include_cost), in normalised
        coordinates. Each design entry is its axis value (kind-aware, see
        _axis_value).
        """
        z_design_raw = np.array(
            [self._axis_value(name) for name in self.z_names], dtype=float
        )
        c_raw = None
        if self.include_cost:
            c_raw = float(self.model.variables["net_present_cost"].solution.sum())
        return self._to_explore_coords(z_design_raw, c_raw)

    @property
    def z_star_explore(self) -> np.ndarray:
        """Baseline point z* in ORACLE polytope coordinates.

        Uses the baseline design vector frozen in __init__ (the fmax LPs
        overwrite the loaded solution). The baseline cost is exactly C*, so the
        cost coordinate (when include_cost) is exactly 0.
        """
        return self._to_explore_coords(self._z_star_design_raw, self.c_star)

    def _compute_dataset_hash(self) -> str:
        """SHA-256-prefix digest of the dataset folder's contents.

        Used to invalidate the fmax cache when the input data has changed.
        Walks the dataset folder recursively in sorted relative-path order,
        feeding both filenames and file contents into the digest, and skips
        dotfiles (e.g. macOS `.DS_Store`). Truncated to 16 hex chars; the
        2^-64 collision probability is fine for a workflow-internal sanity
        check.

        NOTE: this only catches changes to files INSIDE the dataset folder.
        Edits to `config.json`, the zen-garden source code, or solver
        options will silently NOT invalidate the cache — out of scope.
        """
        root = Path(self.optimization_setup.analysis.dataset).resolve()
        h = hashlib.sha256()
        for path in sorted(
            p for p in root.rglob("*")
            if p.is_file() and not any(
                part.startswith(".") for part in p.relative_to(root).parts
            )
        ):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
        return h.hexdigest()[:16]

    def compute_fmax_normalization(self) -> None:
        """Compute U_g* = max z_g over the near-optimal polytope, per z-axis.

        Solves one LP per axis: same model state as the baseline, plus the
        near-optimality cost cap (already added by setup()), with the objective
        replaced by max(axis value) (kind-aware, see _axis_linexpr). U_g*
        serves dually as (a) the per-axis upper bound of the initial outer
        approximation and (b) the normalisation denominator z_g / U_g*.

        Cache file layout (JSON):
            {"dataset_hash": "<sha256-prefix>",
             "entries": {
                 "<axis_name>": {"members": [<sorted tech names>],
                                 "U_star":   <float>},
                 ...
             }}
        ``dataset_hash`` is the digest of the dataset folder's contents (see
        _compute_dataset_hash); if it changes, the entire cache is discarded
        because every U_g* may now be wrong. ``members`` is the membership
        the LP was solved against; if a current axis's sorted members differ
        from the cached entry, that one axis is recomputed.

        Legacy cache files written before this change (flat
        ``{tech_name: U_star_float}``) are read transparently — each entry
        is treated as a singleton with implicit ``members=[tech_name]``, no
        dataset-hash check — and rewritten in the new format on save.

        Sets self.u_star, self._u_tilde, self._offset. Call once, after
        setup() and before setup_projection_model().
        """
        dataset_name = Path(self.optimization_setup.analysis.dataset).name
        # Cache lives next to the run output folders, as a sibling of
        # folder_output (e.g. outputs/oracle_cache/). Deriving it from
        # folder_output keeps it (a) shared across every run writing into the
        # same outputs/ directory and (b) portable across machines — no
        # hardcoded absolute path.
        cache_dir = (
            Path(self.optimization_setup.analysis.folder_output).parent
            / "oracle_cache"
            / dataset_name
        )
        cache_file = cache_dir / f"fmax_eps{self.epsilon}.json"

        # --- Dataset content hash (integrity check; catches dataset edits) ---
        current_hash = self._compute_dataset_hash()

        # --- Load cache, detecting new vs legacy format ---
        # cached_entries: axis_name -> {"members": sorted list[str], "U_star": float}
        cached_entries: dict[str, dict] = {}
        cache_was_legacy = False
        if cache_file.exists():
            with open(cache_file) as f:
                raw_cache = json.load(f)
            is_new_format = (
                isinstance(raw_cache, dict)
                and isinstance(raw_cache.get("dataset_hash"), str)
                and isinstance(raw_cache.get("entries"), dict)
            )
            if is_new_format:
                if raw_cache["dataset_hash"] != current_hash:
                    logging.warning(
                        f"MGA oracle fmax: dataset content changed since cache "
                        f"was written (cached_hash={raw_cache['dataset_hash']}, "
                        f"current_hash={current_hash}); discarding all "
                        f"{len(raw_cache['entries'])} cached entries."
                    )
                    # leave cached_entries empty -> recompute everything
                else:
                    for axis_name, entry in raw_cache["entries"].items():
                        if (
                            isinstance(entry, dict)
                            and isinstance(entry.get("members"), list)
                            and "U_star" in entry
                        ):
                            cached_entries[axis_name] = {
                                "members": sorted(str(m) for m in entry["members"]),
                                "U_star": float(entry["U_star"]),
                            }
                    logging.info(
                        f"MGA oracle fmax: loaded cache {cache_file} "
                        f"({len(cached_entries)} entries, dataset_hash OK)."
                    )
            elif isinstance(raw_cache, dict) and raw_cache and all(
                isinstance(k, str) and isinstance(v, (int, float))
                for k, v in raw_cache.items()
            ):
                # Legacy flat format: {tech_name: U_star_float}. Implicit
                # singleton membership. No dataset-hash check on legacy files.
                cache_was_legacy = True
                for axis_name, u_val in raw_cache.items():
                    cached_entries[axis_name] = {
                        "members": [axis_name],
                        "U_star": float(u_val),
                    }
                logging.info(
                    f"MGA oracle fmax: loaded legacy cache {cache_file} "
                    f"({len(cached_entries)} entries); will upgrade to new "
                    f"format on save."
                )
            else:
                logging.warning(
                    f"MGA oracle fmax: cache file {cache_file} has unrecognised "
                    f"structure; ignoring it (computing fresh)."
                )
        else:
            logging.info(f"MGA oracle fmax: no cache at {cache_file}; computing fresh.")

        # --- Decide reuse vs. recompute per axis ---
        members_by_name = {name: members for name, members in self.z_groups}
        matched: list[str] = []
        recompute_missing: list[str] = []
        recompute_membership_changed: list[str] = []
        for name in self.z_names:
            current_sorted = sorted(members_by_name[name])
            if name not in cached_entries:
                recompute_missing.append(name)
            elif cached_entries[name]["members"] != current_sorted:
                recompute_membership_changed.append(name)
            else:
                matched.append(name)
        # Unused entries: cached axes the current run doesn't reference. We
        # preserve them in the rewritten file (legacy behaviour) — they don't
        # hurt anything and may be useful next run.
        unused = sorted(set(cached_entries.keys()) - set(self.z_names))

        if matched:
            logging.info(
                f"MGA oracle fmax: {len(matched)}/{self.n_z} axes matched "
                f"from cache: {matched}"
            )
        if recompute_missing:
            logging.info(
                f"MGA oracle fmax: {len(recompute_missing)}/{self.n_z} axes "
                f"missing from cache (will compute): {recompute_missing}"
            )
        if recompute_membership_changed:
            logging.info(
                f"MGA oracle fmax: {len(recompute_membership_changed)}/{self.n_z} "
                f"axes have membership changed since cache was written "
                f"(will recompute): {recompute_membership_changed}"
            )
        if unused:
            logging.info(
                f"MGA oracle fmax: {len(unused)} cache entries are unused by "
                f"this run (preserved in the file): {unused}"
            )

        # --- LP loop for axes that need recompute ---
        for name in recompute_missing + recompute_membership_changed:
            members = members_by_name[name]
            kind = self.axis_kind[name]
            # Objective: maximise the axis's value over the near-optimal space.
            # Kind-aware (see _axis_linexpr): sum of capacity_addition for a
            # tech axis, duration-weighted flow_import for a carrier axis.
            obj = self._axis_linexpr(name)
            self.model.add_objective(obj, sense="max", overwrite=True)
            t0 = time.time()
            self.optimization_setup.solve()
            elapsed = time.time() - t0
            if not self.optimization_setup.optimality:
                tc = self.model.termination_condition
                if kind == "carrier_import":
                    hint = (
                        "carrier-import axis; an 'unbounded' status means this "
                        "carrier can be imported at zero/near-zero price within "
                        "the cost budget. Members: " + repr(members)
                    )
                elif len(members) > 1:
                    hint = (
                        "lumped tech group; one of its members may be the "
                        "culprit. Members: " + repr(members)
                    )
                else:
                    hint = "single technology"
                raise RuntimeError(
                    f"MGA oracle fmax: LP for axis {name!r} did not solve to "
                    f"optimality (termination = {tc!r}). An 'unbounded' status "
                    f"means this axis has no finite near-optimal maximum — "
                    f"exclude the offending member ({hint})."
                )
            u_i = self._axis_value(name)
            cached_entries[name] = {
                "members": sorted(members),
                "U_star": u_i,
            }
            logging.info(
                f"MGA oracle fmax: U*[{name}] = {u_i:.6g} ({kind}, LP took "
                f"{elapsed:.1f} s, {len(members)} "
                f"member{'s' if len(members) != 1 else ''})."
            )

        # --- Save cache in NEW format (upgrades legacy in place on first run) ---
        cache_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "dataset_hash": current_hash,
            "entries": {
                name: {
                    "members": list(entry["members"]),
                    "U_star": entry["U_star"],
                }
                for name, entry in cached_entries.items()
            },
        }
        with open(cache_file, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        upgrade_note = " (legacy format upgraded)" if cache_was_legacy else ""
        logging.info(f"MGA oracle fmax: cache written to {cache_file}{upgrade_note}.")

        self.u_star = np.array(
            [cached_entries[n]["U_star"] for n in self.z_names], dtype=float
        )

        # Sanity A7: every U_g* must be >= the baseline z_g* (it is by
        # construction the max over a polytope that contains z*).
        z_star = self._z_star_design_raw
        tol = 1e-6 * (np.abs(self.u_star) + 1.0)
        bad = self.u_star < z_star - tol
        if bad.any():
            i = int(np.argmax(z_star - self.u_star))
            raise RuntimeError(
                f"MGA oracle fmax: U*[{self.z_names[i]}] = {self.u_star[i]:.6g} "
                f"< baseline z*[{self.z_names[i]}] = {z_star[i]:.6g}. The "
                f"fmax LP is inconsistent with the baseline solve."
            )
        # A zero/negative U_g* cannot serve as a normalisation denominator.
        if (self.u_star <= 0).any():
            i = int(np.argmin(self.u_star))
            raise RuntimeError(
                f"MGA oracle fmax: U*[{self.z_names[i]}] = {self.u_star[i]:.6g} "
                f"<= 0 — cannot normalise. Exclude this axis or its members."
            )

        # Augmented scale/offset: design axes use U_i* with offset 0; the cost
        # axis (when include_cost) uses the slack range eps*C* with offset C*.
        u = list(self.u_star)
        off = [0.0] * self.n_z
        if self.include_cost:
            u.append(self.epsilon * self.c_star)
            off.append(self.c_star)
        self._u_tilde = np.array(u, dtype=float)
        self._offset = np.array(off, dtype=float)
        logging.info(
            f"MGA oracle fmax: ready. U* range [{self.u_star.min():.4g}, "
            f"{self.u_star.max():.4g}] (raw z units); "
            f"cost axis scale eps*C* = {self.epsilon * self.c_star:.6g}"
            + (" (active)" if self.include_cost else " (unused, include_cost=False)")
            + "."
        )

    def build_initial_outer_approximation(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct (A0, b0) for the initial outer polytope, in z_groups order.

        The bounds come from a SINGLE source: the per-axis fmax maxima U_i*
        (one LP per axis, see compute_fmax_normalization). Row blocks, in order
        (n_z = number of exploration axes; coordinates are always normalised):

          1. n_z non-negativity rows  -z_i <= 0   (z_i >= 0: capacity_addition
             and flow_import are both non-negative, so every axis value is too).
          2. n_z fmax upper-bound rows  z_i <= U_i*. Provably valid (U_i* is by
             construction the maximum of z_i over the near-optimal space) and
             tight. This is the whole outer approximation on the design axes.
          3. Cost rows (include_cost only): C <= (1+eps)*C* and -C <= -C*.

        The near-optimal cost budget c^T x <= (1+eps)*C* itself lives in the
        energy-system model (added once by setup()), NOT in this polytope, so
        it is enforced on every projection solve regardless of these rows.

        The assembled raw polytope is then mapped to normalised coordinates:
        with z_raw = offset + diag(U_tilde) z_norm, each raw row a^T z_raw <= b
        becomes (a o U_tilde)^T z_norm <= b - a^T offset (see _to_explore_coords).
        In normalised coordinates the fmax rows are simply z_norm,i <= 1 and the
        cost-upper row is C_norm <= 1.

        Containment of z* is asserted before returning; a violation raises.

        Returns:
            (A0, b0): np.ndarray of shape (n_rows, n_explore) and (n_rows,),
            in ORACLE (normalised) polytope coordinates.
        """
        n_z = self.n_z

        # --- Raw design rows: non-negativity + fmax upper bounds ---
        #   -z_i <= 0           (non-negativity)
        #    z_i <= U_i*        (fmax box)
        A0 = np.vstack([-np.eye(n_z), np.eye(n_z)])
        b0 = np.concatenate([np.zeros(n_z), self.u_star])

        # --- Cost coordinate: 2 rows  C <= (1+eps)C*  and  -C <= -C*  ---
        if self.include_cost:
            # widen every existing row with a zero cost column
            A0 = np.hstack([A0, np.zeros((A0.shape[0], 1))])
            cost_up = np.zeros((1, n_z + 1)); cost_up[0, n_z] = 1.0
            cost_lo = np.zeros((1, n_z + 1)); cost_lo[0, n_z] = -1.0
            A0 = np.vstack([A0, cost_up, cost_lo])
            b0 = np.concatenate(
                [b0, [(1.0 + self.epsilon) * self.c_star], [-self.c_star]]
            )

        # --- Normalisation transform (always on for ORACLE) ---
        # Each raw row  a^T z_raw <= b  with z_raw = offset + diag(U_tilde) z_norm
        # becomes  (a o U_tilde)^T z_norm <= b - a^T offset.
        b0 = b0 - A0 @ self._offset       # uses raw A0 — must precede scaling
        A0 = A0 @ np.diag(self._u_tilde)

        # --- Sanity assert: A0 @ z*_explore <= b0  (containment) ---
        z_star = self.z_star_explore
        lhs = A0 @ z_star
        abs_tol = 1e-6 * (np.abs(b0) + 1.0)
        violations = lhs > b0 + abs_tol
        if violations.any():
            i = int(np.argmax(lhs - b0))
            raise RuntimeError(
                f"MGA initial outer approximation violates containment of z*: "
                f"row {i}: lhs={lhs[i]:.6g} > rhs={b0[i]:.6g} "
                f"(slack {lhs[i] - b0[i]:.3g}). Row blocks: [0,{n_z}) "
                f"non-negativity, [{n_z},{2 * n_z}) fmax upper bounds, "
                f"remaining rows cost upper/lower."
            )

        # --- Diagnostic logging ---
        n_cost_rows = 2 if self.include_cost else 0
        n_tech = sum(1 for k in self.axis_kind.values() if k == "tech_capacity")
        n_carrier = self.n_z - n_tech
        logging.info(
            f"MGA outer approximation [include_cost={self.include_cost}]: "
            f"n_explore={A0.shape[1]} (n_z={n_z} axes = {n_tech} tech + "
            f"{n_carrier} carrier{', + 1 cost' if self.include_cost else ''}), "
            f"rows={A0.shape[0]} = {n_z} non-neg + {n_z} fmax upper "
            f"+ {n_cost_rows} cost. Coordinates NORMALISED (z/U*): design axes "
            f"dimensionless with near-optimal range [0,1]"
            + ("; cost axis = (C-C*)/(eps*C*), range [0,1]." if self.include_cost
               else ".")
        )
        return A0, b0

    def _solve_and_postprocess(self, label: str) -> None:
        """Solve current model state and write a Postprocess folder labelled `<base>_<label>`."""
        self.optimization_setup.solve()
        if not self.optimization_setup.optimality:
            raise RuntimeError(
                f"MGA solve failed (label={label}): "
                f"termination = {self.model.termination_condition}"
            )
        base_name = self.postprocess_ctx["model_name"]
        Postprocess(
            self.optimization_setup,
            scenarios=self.postprocess_ctx["scenarios"],
            subfolder=self.postprocess_ctx["subfolder"],
            model_name=f"{base_name}_{label}",
            scenario_name=self.postprocess_ctx["scenario_name"],
            param_map=self.postprocess_ctx["param_map"],
        )

    # ------------------------------------------------------------------
    # random_directions-mode callback
    # ------------------------------------------------------------------

    def support_function(self, direction: np.ndarray):
        """Callback for pyoNearOpt random_directions.

        Solves max d^T z over Z_eps via min (-d)^T z. Returns (z_feas, d @ z_feas).
        """
        if direction.shape != (self.n_z,):
            raise ValueError(
                f"support_function: expected shape ({self.n_z},), got {direction.shape}"
            )

        # Build a full-coord weight DataArray inline: zero everywhere except
        # on member techs, where each carries -direction[g] for its group g
        # (sign flip because we maximise direction^T z over Z_eps via min
        # (-direction)^T z). For a lumped group the same -direction[g] is
        # placed on every member, so (w * cap_add).sum() correctly evaluates
        # sum_g -direction[g] * (sum_{m in g} z_m) = -direction · z_grouped.
        tech_coord = self.cap_add.coords["set_technologies"]
        w = xr.DataArray(
            np.zeros(tech_coord.size),
            dims=("set_technologies",),
            coords={"set_technologies": tech_coord},
        )
        for g_idx, (_name, members) in enumerate(self.z_groups):
            for m in members:
                w.loc[m] = -float(direction[g_idx])

        obj = (w * self.cap_add).sum()
        self.model.add_objective(obj, sense="min", overwrite=True)

        label = f"random_dir_iter_{self._iter_count}"
        logging.info(f"MGA random_directions: starting iteration {self._iter_count}")
        self._solve_and_postprocess(label)

        z_feas = self._extract_z()
        support_value = float(direction @ z_feas)
        logging.info(
            f"MGA random_directions iter {self._iter_count}: "
            f"support_value = {support_value:.4g}"
        )
        self._iter_count += 1
        return z_feas, support_value

    # ------------------------------------------------------------------
    # oracle-mode setup and callback
    # ------------------------------------------------------------------

    def setup_projection_model(self) -> None:
        """Add the L-infinity projection model to the linopy model.

        The per-axis dim name self._z_dim is chosen conditionally:
          - ``set_technologies`` when every z-axis is a singleton tech (name ==
            tech name): the proven vectorised path, byte-identical to today.
          - ``mga_z_axis`` otherwise (tech lumps and/or carrier axes), because
            lumped/carrier names are not valid coord values on cap_add's
            ``set_technologies`` dim.

        Projection-equality construction also branches:
          - No carrier axis: ONE vectorised constraint mga_oracle_proj_eq
            (Sx - delta == trial), Sx built from cap_add via a selector matrix
            (tech lumps) or a plain .sel (pure singletons). Unchanged.
          - Any carrier axis: capacity_addition and flow_import cannot share a
            single vectorised selector, so one named scalar equality PER axis,
            mga_oracle_proj_eq_axis{i}: axis_expr_i - delta[i] == trial_i, with
            axis_expr_i from the kind-aware _axis_linexpr. delta stays a vector
            for the (vectorised) t-constraints.

        Coordinates are ALWAYS normalised: d_scale_g = 1/U_g* in the t-bounds,
        so min t = max_g |delta_g|/U_g* = the normalised L-inf distance.

        Variables: mga_oracle_delta (vector on _z_dim), mga_oracle_t (scalar),
        and mga_oracle_delta_cost (scalar, include_cost only). The cost track
        (mga_oracle_proj_eq_cost + t-bounds, c_scale = 1/(eps*C*)) is unchanged.

        Call exactly once before the ORACLE loop.
        """
        # Pure singleton-tech runs keep the set_technologies dim (byte-identical
        # proven path); anything else (tech lumps or carrier axes) uses a fresh
        # dim, since those axis names are not cap_add tech coords.
        self._z_dim = "mga_z_axis" if self._has_lumped_groups else "set_technologies"

        z_coord = xr.DataArray(
            np.array(self.z_names),
            dims=self._z_dim,
            coords={self._z_dim: self.z_names},
        )

        self.delta = self.model.add_variables(
            coords=[z_coord], name="mga_oracle_delta",
            lower=-np.inf, upper=np.inf,
        )

        # t_var needs a (trivial) dimension so that ZEN-garden's Postprocess
        # can convert it to a DataFrame. A truly scalar variable would crash
        # postprocess.save_var with "cannot convert a scalar to a DataFrame".
        t_coord = xr.DataArray(
            np.array([0]),
            dims="mga_oracle_scalar_dim",
            coords={"mga_oracle_scalar_dim": [0]},
        )
        self.t_var = self.model.add_variables(
            coords=[t_coord], name="mga_oracle_t",
            lower=0.0,
        )

        if self.has_carrier_axis:
            # --- Per-axis named equality constraints (mixed variable kinds) ---
            # One scalar constraint per axis: axis_expr_i - delta[i] == trial_i.
            # RHS updated per iteration in find_nearest_point.
            for i, name in enumerate(self.z_names):
                expr = self._axis_linexpr(name)
                self.model.add_constraints(
                    expr - self.delta.sel({self._z_dim: name}) == 0.0,
                    name=f"mga_oracle_proj_eq_axis{i}",
                )
        else:
            # --- Vectorised path (tech-only; proven, byte-identical) ---
            if self._has_lumped_groups:
                # 2-D selector matrix: 1.0 at (member_tech, axis_name) for each
                # group's members. Multiplying by cap_add and summing reduces it
                # to a LinearExpr on mga_z_axis, one entry per axis.
                all_techs = list(self.cap_add.coords["set_technologies"].values)
                selector = xr.DataArray(
                    np.zeros((len(all_techs), self.n_z)),
                    dims=("set_technologies", self._z_dim),
                    coords={"set_technologies": all_techs,
                            self._z_dim: self.z_names},
                )
                for _g_idx, (name, members) in enumerate(self.z_groups):
                    for m in members:
                        selector.loc[
                            {"set_technologies": m, self._z_dim: name}
                        ] = 1.0
                Sx = (selector * self.cap_add).sum(
                    ["set_technologies", "set_capacity_types",
                     "set_location", "set_time_steps_yearly"]
                )
            else:
                # Pure singletons: dim IS set_technologies, byte-identical.
                Sx = self.cap_add.sel(set_technologies=self.z_techs).sum(
                    ["set_capacity_types", "set_location", "set_time_steps_yearly"]
                )
            zero_trial = xr.DataArray(
                np.zeros(self.n_z),
                dims=self._z_dim,
                coords={self._z_dim: self.z_names},
            )
            self.model.add_constraints(
                Sx - self.delta == zero_trial, name="mga_oracle_proj_eq"
            )

        # Per-axis L-inf scaling: d_scale_g = 1/U_g* (normalisation always on),
        # so min t yields t* = max_g |delta_g|/U_g* = the normalised L-inf
        # distance. Vectorised over the whole delta vector in both paths.
        d_scale = xr.DataArray(
            1.0 / self.u_star,
            dims=self._z_dim,
            coords={self._z_dim: self.z_names},
        )
        self.model.add_constraints(
            d_scale * self.delta - self.t_var <= 0, name="mga_oracle_t_pos"
        )
        self.model.add_constraints(
            -(d_scale * self.delta) - self.t_var <= 0, name="mga_oracle_t_neg"
        )

        # --- Cost-coordinate projection track (include_cost only) ---
        # delta_cost absorbs C(x) - trial_C; the SHARED t_var makes t* the
        # L-inf distance over all n_z+1 normalised axes at once.
        if self.include_cost:
            self.delta_cost = self.model.add_variables(
                coords=[t_coord], name="mga_oracle_delta_cost",
                lower=-np.inf, upper=np.inf,
            )
            cost_expr = self.optimization_setup.energy_system.rules.objective_total_cost(
                self.model
            )
            zero_cost = xr.DataArray(
                np.zeros(1),
                dims="mga_oracle_scalar_dim",
                coords={"mga_oracle_scalar_dim": [0]},
            )
            self.model.add_constraints(
                cost_expr - self.delta_cost == zero_cost,
                name="mga_oracle_proj_eq_cost",
            )
            c_scale = 1.0 / (self.epsilon * self.c_star)
            self.model.add_constraints(
                c_scale * self.delta_cost - self.t_var <= 0,
                name="mga_oracle_t_pos_cost",
            )
            self.model.add_constraints(
                -(c_scale * self.delta_cost) - self.t_var <= 0,
                name="mga_oracle_t_neg_cost",
            )

        logging.info(
            f"MGA oracle: projection model added (n_z = {self.n_z} axes on "
            f"dim {self._z_dim!r}, proj-eq path = "
            f"{'per-axis' if self.has_carrier_axis else 'vectorised'}, "
            f"include_cost = {self.include_cost})"
        )

    def find_nearest_point(self, trial_point: np.ndarray):
        """Callback for pyoNearOpt ORACLE.

        trial_point arrives in ORACLE polytope coordinates (always normalised;
        augmented with a cost coordinate as the last entry when include_cost).
        Returns (z_feas, dist, mu_cut, b_cut, flag) in the SAME coordinates;
        mu_cut is L2-normalised and b_cut = mu_cut @ z_feas.

        Coordinate algebra:
            trial_raw = offset + trial_norm * U_tilde         (proj-eq RHS)
            dist      = t*  — the projection LP minimises the NORMALISED
                        L-inf distance directly (scaled t-constraints), so no
                        post-hoc recomputation is needed or correct.
            mu_norm   = mu_raw o U_tilde   (mu_raw = dual of the raw proj eq)
            b_cut     = mu_norm @ z_feas_norm
        The affine cost offset C* cancels in the cut, so only U_tilde scales mu.
        """
        if trial_point.shape != (self.n_explore,):
            raise ValueError(
                f"find_nearest_point: expected shape ({self.n_explore},), "
                f"got {trial_point.shape}"
            )

        # --- Design axes: trial point -> raw coordinates, set proj-eq RHS ---
        # trial_design_raw = trial_norm * U_g* (the design offset is 0).
        trial_design_raw = trial_point[:self.n_z] * self.u_star
        if self.has_carrier_axis:
            # Per-axis named scalar equality: set each RHS to its raw trial value.
            for i in range(self.n_z):
                self.model.constraints[f"mga_oracle_proj_eq_axis{i}"].rhs = float(
                    trial_design_raw[i]
                )
        else:
            trial_da = xr.DataArray(
                trial_design_raw,
                dims=self._z_dim,
                coords={self._z_dim: self.z_names},
            )
            self.model.constraints["mga_oracle_proj_eq"].rhs = trial_da

        # --- Cost axis: trial point -> raw cost, set cost proj-eq RHS ---
        if self.include_cost:
            trial_c = float(trial_point[self.n_z])
            trial_c_raw = self.c_star + trial_c * self.epsilon * self.c_star
            trial_c_da = xr.DataArray(
                np.array([trial_c_raw]),
                dims="mga_oracle_scalar_dim",
                coords={"mga_oracle_scalar_dim": [0]},
            )
            self.model.constraints["mga_oracle_proj_eq_cost"].rhs = trial_c_da

        # Objective: min t. Sum collapses the trivial mga_oracle_scalar_dim
        # back to a scalar expression for the objective.
        self.model.add_objective(self.t_var.sum(), sense="min", overwrite=True)

        label = f"oracle_iter_{self._iter_count}"
        logging.info(
            f"MGA oracle: starting iteration {self._iter_count}, "
            f"||trial_point||_2 = {np.linalg.norm(trial_point):.4g}"
        )
        self._solve_and_postprocess(label)

        # z_feas in polytope coordinates (already normalised/augmented).
        z_feas = self._extract_z()

        # dist = t* = the normalised L-inf projection distance.
        dist = float(self.t_var.solution.values[0])

        # --- Cutting hyperplane: dual of the projection equality, rescaled ---
        # mu_raw = dual of the raw proj eq (sign convention matches scipy
        # linprog eqlin.marginals). mu_norm = mu_raw o U_g*.
        if self.has_carrier_axis:
            mu_design_raw = np.array(
                [
                    float(
                        self.model.constraints[f"mga_oracle_proj_eq_axis{i}"]
                        .dual.values
                    )
                    for i in range(self.n_z)
                ],
                dtype=float,
            )
        else:
            mu_design_raw = (
                self.model.constraints["mga_oracle_proj_eq"].dual
                .sel({self._z_dim: self.z_names})
                .values
            )
        mu_cut = np.asarray(mu_design_raw, dtype=float) * self.u_star

        if self.include_cost:
            mu_c_raw = float(
                self.model.constraints["mga_oracle_proj_eq_cost"].dual.values[0]
            )
            mu_c = mu_c_raw * self.epsilon * self.c_star
            mu_cut = np.append(mu_cut, mu_c)

        # L2-normalise the (already coordinate-scaled) cut normal, then take
        # b_cut from the SAME normalised mu_cut so the pair stays consistent.
        scale = np.linalg.norm(mu_cut, ord=2)
        if scale > 1e-4:
            mu_cut = mu_cut / scale
        b_cut = float(mu_cut @ z_feas)

        logging.info(
            f"MGA oracle iter {self._iter_count}: dist = {dist:.4g} "
            f"(normalised L-inf), |mu|_max = {np.max(np.abs(mu_cut)):.4g}, "
            f"b_cut = {b_cut:.4g}"
        )
        self._iter_count += 1
        return z_feas, dist, mu_cut, b_cut, 0


# ----------------------------------------------------------------------
# Event handler (mode dispatcher)
# ----------------------------------------------------------------------

@EventPublisher.register(Event.after_solve)
def run_mga(*args, **kwargs):
    """Entry point invoked by EventPublisher after the baseline solve."""
    mode = config.get("mode", "weights")
    epsilon = config["epsilon"]
    exclude_techs = config.get("exclude_techs", [])
    include_techs = config.get("include_techs", [])
    include_carrier_imports = config.get("include_carrier_imports", [])

    # Oracle exploration-coordinate options. ORACLE always normalises by the
    # per-axis fmax maxima — there is no longer a normalization knob.
    ora_cfg = config.get("oracle", {})
    include_cost = bool(ora_cfg.get("include_cost", False))
    if "normalization" in ora_cfg:
        logging.info(
            "MGA oracle: the 'normalization' config key is deprecated and "
            "ignored — normalization is always on for ORACLE."
        )

    optimization_setup = kwargs["optimization_setup"]
    postprocess_ctx = {
        "scenarios": kwargs["scenarios"],
        "subfolder": kwargs["subfolder"],
        "model_name": kwargs["model_name"],
        "scenario_name": kwargs["scenario_name"],
        "param_map": kwargs["param_map"],
    }
    logging.info(
        f"MGA plugin: mode = {mode!r}, epsilon = {epsilon}, "
        f"n_exclude = {len(exclude_techs)}, n_include = {len(include_techs)}, "
        f"n_carrier_imports = {len(include_carrier_imports)}"
    )

    mga = MGA(
        optimization_setup, epsilon, postprocess_ctx,
        exclude_techs=exclude_techs,
        include_techs=include_techs,
        include_carrier_imports=include_carrier_imports,
        include_cost=include_cost,
    )
    mga.setup()

    if mode == "weights":
        iterations = config.get("iterations", [])
        if not iterations:
            logging.warning("MGA plugin: weights mode but no iterations configured; skipping.")
            return
        for i, iteration in enumerate(iterations):
            if "weights" not in iteration:
                raise KeyError(f"MGA iteration {i} missing 'weights' key")
            weights = iteration["weights"]
            if not isinstance(weights, dict):
                raise TypeError(
                    f"MGA iteration {i}: 'weights' must be a dict, "
                    f"got {type(weights).__name__}"
                )
            mga.run_iteration(weights, i)

    elif mode == "random_directions":
        raise NotImplementedError(
            "MGA random_directions mode was retired when the legacy "
            "capacity-based outer-approximation bounds were removed; the fmax "
            "box (ORACLE-only) is now the sole bound source, and "
            "random_directions never ran the fmax LPs. Use mode='oracle'."
        )

    elif mode == "oracle":
        from pyoNearOpt.exploration_methods.ORACLE import oracle as ORACLEAlgorithm
        from pyoNearOpt.polytope_approximation.approximation_class import approximation
        import pyomo.environ as pyo

        max_iter = ora_cfg.get("max_iterations", 200)
        if "tolerance" not in ora_cfg:
            raise ValueError("MGA oracle: 'tolerance' is required.")
        tol = float(ora_cfg["tolerance"])
        Md_override = ora_cfg.get("Md_override", 1e8)
        t_max_override = ora_cfg.get("t_max_override", None)

        # fmax: solve the n_z auxiliary U_i* LPs (cached on disk) BEFORE the
        # projection model is added and before the ORACLE loop starts. These
        # supply both the normalisation denominators and the initial outer box.
        mga.compute_fmax_normalization()

        # Add projection model to the linopy problem (ONCE).
        mga.setup_projection_model()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Coordinates across variables not equal.*",
                category=UserWarning,
            )
            A0, b0 = mga.build_initial_outer_approximation()
            z_star = mga.z_star_explore
            logging.info(f"MGA oracle: tol = {tol:.3g}")

            name_list = list(mga.z_names) + (
                ["net_present_cost"] if mga.include_cost else []
            )
            poly = approximation(
                A=A0, X=z_star.reshape(1, -1), b=b0,
                name_list=name_list,
                use_bigM=False,
            )
            # Big-M defaults are tuned for normalised toys; loosen for Crystal Ball.
            poly.Md = float(Md_override)
            if t_max_override is not None:
                poly.t_max = float(t_max_override)

        # Pyomo solver for ORACLE's internal polytope MILPs.
        # solver_io="python":  use Gurobi's Python API directly (no LP file IO).
        # manage_env=True:     create+release Gurobi env in the constructor;
        #                      protects single-use academic licenses from hanging.
        # OutputFlag=0:        suppress per-MILP console output (30 iterations
        #                      of Gurobi log spam otherwise).
        pyomo_solver = pyo.SolverFactory(
            "gurobi", solver_io="python", manage_env=True
        )
        pyomo_solver.set_options("OutputFlag=1 MIPGap=0.05")

        algo = ORACLEAlgorithm(
            poly_approx=poly,
            find_nearest_point=mga.find_nearest_point,
            max_iterations=max_iter,
            tol=tol,
            pyomo_solver=pyomo_solver,
            print_lv=1,
        )

        # Summary folder lands inside folder_output, as a sibling of the
        # per-iteration Postprocess folders. postprocess_ctx["subfolder"] is
        # only a relative sub-path (empty for non-scenario runs), so using it
        # alone would resolve against the process cwd instead.
        out = (
            Path(optimization_setup.analysis.folder_output)
            / f"{postprocess_ctx['model_name']}_oracle_summary"
        )
        out.mkdir(parents=True, exist_ok=True)
        df = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Coordinates across variables not equal.*",
                    category=UserWarning,
                )
                df = algo.refine_approximations()
        finally:
            # Persist polytope (and diagnostics if available) even if
            # refine_approximations raised mid-way (Big-M violation, infeasible
            # projection, etc.). This avoids losing 25 of 30 successful
            # iterations because iteration 26 failed.
            np.savez(
                out / "polytope.npz",
                A=poly.A, b=poly.b, X=poly.X,
                name_list=np.array(poly.name_list),
            )
            if df is not None:
                df.to_csv(out / "diagnostics.csv", index=False)
            # Convergence status only meaningful if refine_approximations
            # completed normally (df is not None means it returned a result).
            if df is not None:
                final_dist = float(df["max_min_distance"].iloc[-1])
                n_iters_done = len(df)
                converged = final_dist <= tol
                if converged:
                    logging.info(
                        f"MGA oracle: CONVERGED in {n_iters_done} iterations. "
                        f"Final max_min_distance = {final_dist:.4g} <= tol = {tol:.4g}."
                    )
                else:
                    logging.warning(
                        f"MGA oracle: did NOT converge after {n_iters_done} iterations. "
                        f"Final max_min_distance = {final_dist:.4g}, tol = {tol:.4g}. "
                        f"Increase max_iterations or relax tolerance."
                    )
            else:
                logging.warning(
                    "MGA oracle: refine_approximations did not return a result "
                    "(likely raised mid-iteration). See traceback above."
                )
            logging.info(f"MGA oracle: artifacts saved to {out}")

    else:
        raise ValueError(
            f"Unknown MGA mode: {mode!r}. "
            f"Expected 'weights', 'random_directions', or 'oracle'."
        )

    logging.info("MGA plugin: complete.")
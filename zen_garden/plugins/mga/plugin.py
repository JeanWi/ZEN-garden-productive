"""
MGA (Modeling to Generate Alternatives) plugin for ZEN-garden.

Registers an `after_solve` event handler that, after the baseline optimization
has been solved, adds a near-optimality cost constraint and re-solves the
problem under one of three exploration modes selected via config:

    * "weights" (existing): user-provided list of per-iteration weight dicts;
      each iteration solves min sum_i w_i * cap_add_i.
    * "random_directions" (new): uniform sampling of directions on the unit
      hypersphere; per direction, one solve max d^T z over Z_eps. Delegates
      the loop to pyoNearOpt.exploration_methods.random.random_directions.
    * "oracle" (new): ORACLE algorithm from Turan, Moret, Bardow (2026).
      Iteratively refines inner/outer polytope approximations of Z_eps via
      L-infinity projections of trial points. Delegates the loop to
      pyoNearOpt.exploration_methods.ORACLE.oracle.

The initial outer polytope (oracle + random_directions modes) is derived
directly from the model: per-tech upper bounds from `capacity_addition_max`
(Tier 1) plus one cost-cap half-space with per-tech `min`-under-approximated
unit cost (Tier 2). No tuneable knobs — see `MGA.build_initial_outer_approximation`.

Each iteration is written to disk as a sibling sub-solution next to the
baseline, using Postprocess with a modified model_name.

Configuration (via the "plugins.mga" block in config.json):
    epsilon (float): near-optimality slack. Default 0.1.
    mode (str): "weights" | "random_directions" | "oracle". Default "weights".
    exclude_techs (list[str]): technologies excluded from z (the exploration
        vector). Used in random_directions and oracle modes only; ignored in
        weights mode (where excluded techs simply default to weight 0).
    iterations (list[dict]): used only in weights mode. One entry per MGA
        iteration with a "weights": {tech_name: float} dict.
    random_directions (dict): used only in random_directions mode. Keys:
        n_iterations (int), seed (int|None).
    oracle (dict): used only in oracle mode. Keys:
        max_iterations (int), tolerance (float, REQUIRED — no default),
        Md_override (float), t_max_override (float|None).
        Solver is hardcoded to Gurobi.
"""

import logging
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
    "iterations": [],
    "random_directions": {},
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

    def __init__(self, optimization_setup, epsilon, postprocess_ctx, exclude_techs=None):
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
                exploration vector z. Used in random_directions and oracle
                modes; ignored in weights mode.
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

        # Build the canonical exploration-tech list (z_techs) from the full
        # set_technologies coord, with optional exclusions. This is the
        # SINGLE source of truth for the ordering of w, z, mu, name_list
        # everywhere downstream.
        all_techs = list(self.cap_add.coords["set_technologies"].values)
        excluded = set(exclude_techs or [])
        unknown = excluded - set(all_techs)
        if unknown:
            raise KeyError(
                f"exclude_techs contains unknown technologies: {sorted(unknown)}"
            )
        self.z_techs = [t for t in all_techs if t not in excluded]
        self.n_z = len(self.z_techs)

        # Iteration counter for labelling Postprocess folders in the new modes.
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

    def _extract_z(self) -> np.ndarray:
        """Read z = S*x from the most recent solve.

        Returns a 1D ndarray of length n_z in the canonical self.z_techs order.
        xarray's .sel(coord=[list]) preserves the order of the list, so a
        single vectorised sel + sum is equivalent to a per-tech loop but cheaper.
        """
        return (
            self.cap_add.solution
            .sel(set_technologies=self.z_techs)
            .sum(["set_capacity_types", "set_location", "set_time_steps_yearly"])
            .values
        )

    def build_initial_outer_approximation(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct (A0, b0) for the initial outer polytope on z_techs order.

        Row blocks (in order):
          1. n_z non-negativity rows  -z_i <= 0
          2. Tier 1 per-tech upper bounds from two independent model sources;
             a tech may receive a row from either, both, or neither (the
             polytope intersection naturally keeps the tighter):
               2a. `capacity_addition_max[tech, cap_type]` aggregated over
                   (cap_type, loc, year).
               2b. `capacity_limit[tech, cap_type, loc, year]` summed over
                   (cap_type, loc, year). Valid because, per
                   constraint_technology_lifetime + capacity_limit, every
                   capacity_addition[h,c,p,y] <= capacity_limit[h,c,p,y]; the
                   sum runs over ALL years (a short-lifetime tech can retire
                   and re-add, so a single year's limit is not a valid cap).
             For either source a row is emitted only if EVERY model-defined
             (notnull) tuple is finite; one +inf tuple leaves z_i unbounded
             from that source and the row is skipped.
          3. One Tier 2 cost-cap row  tilde_c^T z <= (1+eps) * C*, with
             tilde_c_i = min over (cap_type, loc, year) of the per-tuple
             cost coefficient in the objective. This is a provably valid
             under-approximation: for any feasible x with z = S x,
                 tilde_c_i z_i = tilde_c_i * sum_j x_j <= sum_j c_j x_j
             (since tilde_c_i <= c_j and x_j >= 0), and summing gives
             tilde_c^T z <= c^T x <= (1+eps) C*. Techs without a usable
             capex parameter (e.g. PWA-only conversion, transport with only
             distance-cost) contribute tilde_c_i = 0, which is also valid.

        Containment of z* is asserted before returning; a violation raises.

        Returns:
            (A0, b0): np.ndarray of shape (n_rows, n_z) and (n_rows,).
        """
        es = self.optimization_setup.energy_system
        params = self.optimization_setup.parameters
        n_z = self.n_z

        # --- Tier 1: per-tech aggregate upper bounds from capacity_addition_max ---
        cap_add_max = params.capacity_addition_max  # [set_technologies, set_capacity_types]
        # valid-tuple counts per (tech, cap_type): tuples where the variable exists
        n_valid = self.cap_add.upper.notnull().sum(
            ["set_location", "set_time_steps_yearly"]
        )

        cap_types = list(self.cap_add.coords["set_capacity_types"].values)
        tier1_rows: list[np.ndarray] = []
        tier1_b: list[float] = []
        tier1_techs: list[str] = []
        for idx, tech in enumerate(self.z_techs):
            ub = 0.0
            unbounded = False
            for c in cap_types:
                k = int(n_valid.sel(set_technologies=tech, set_capacity_types=c))
                if k == 0:
                    # This cap_type isn't used by this tech (no valid tuples).
                    # The capacity_addition_max value for it is irrelevant.
                    continue
                m = float(cap_add_max.sel(set_technologies=tech, set_capacity_types=c))
                if not np.isfinite(m) or m == 0.0:
                    # 0 or inf or NaN: the max-cap constraint is masked out at
                    # solve time, so we have no valid bound on this cap_type.
                    unbounded = True
                    break
                ub += m * k
            if unbounded:
                continue
            row = np.zeros(n_z)
            row[idx] = 1.0
            tier1_rows.append(row)
            tier1_b.append(ub)
            tier1_techs.append(tech)
        n_tier1_cam = len(tier1_rows)

        # --- Tier 1b: per-tech aggregate upper bounds from capacity_limit ---
        # capacity_limit bounds the cumulative `capacity` variable. Via
        # constraint_technology_lifetime (capacity[h,c,p,y] = existing +
        # sum_{py in lifetime(y)} capacity_addition[h,c,p,py], with y always in
        # lifetime(y)) and constraint_technology_capacity_limit (capacity <=
        # capacity_limit where finite; capacity_addition == 0 where the limit
        # is already reached), every per-tuple capacity_addition is bounded:
        # capacity_addition[h,c,p,y] <= capacity_limit[h,c,p,y]. Summing over
        # (cap_type, loc, year) gives a valid per-tech UB on z_i. The sum runs
        # over ALL years, not just the last: a short-lifetime tech can add,
        # retire, and re-add, so total horizon additions are bounded by the
        # per-year limits summed, not by any single year's limit. (For a
        # 1-year dataset like cb_small the two coincide.) A row is emitted
        # only if EVERY model-defined (notnull) tuple is finite.
        cap_limit = params.capacity_limit
        for idx, tech in enumerate(self.z_techs):
            sub = cap_limit.sel(set_technologies=tech)
            if not bool(sub.notnull().any()):
                continue  # capacity_limit not defined for this tech
            if bool(np.isinf(sub).any()):
                continue  # >= 1 tuple unbounded -> no valid finite aggregate
            ub = float(sub.sum())  # skipna: sums finite tuples (incl 0), skips NaN
            row = np.zeros(n_z)
            row[idx] = 1.0
            tier1_rows.append(row)
            tier1_b.append(ub)
        n_tier1_clim = len(tier1_rows) - n_tier1_cam

        # --- Tier 2: per-tech min-under-approximated unit cost ---
        dr = float(params.discount_rate)
        ibg = int(es.system.interval_between_years)
        years = list(es.set_time_steps_yearly)
        last_horizon = es.set_time_steps_yearly_entire_horizon[-1]
        # Per-year discount factor, mirroring constraint_net_present_cost.
        discount_factor: dict[int, float] = {}
        for y in years:
            interval = 1 if y == last_horizon else ibg
            discount_factor[y] = sum(
                (1.0 / (1.0 + dr)) ** (ibg * (y - years[0]) + i)
                for i in range(interval)
            )
        # Inverted lifetime range is computed per-tech below (depends on
        # depreciation_time, which is per-tech).
        from zen_garden.model.technology.technology import Technology

        set_conv = set(es.set_conversion_technologies)
        set_stor = set(es.set_storage_technologies)
        set_tran = set(es.set_transport_technologies)

        tilde_c = np.zeros(n_z)
        n_pwa = 0
        n_no_capex = 0
        for idx, tech in enumerate(self.z_techs):
            # per-tech annuity factor
            lt = float(params.depreciation_time.sel(set_technologies=tech))
            if dr != 0.0:
                a_h = ((1 + dr) ** lt * dr) / ((1 + dr) ** lt - 1)
            else:
                a_h = 1.0 / lt
            # per-tech sum_disc_by_py: depends on depreciation_time
            tech_sum_disc: dict[int, float] = {py: 0.0 for py in years}
            for y in years:
                for py in Technology.get_lifetime_range(
                    self.optimization_setup, tech, y, use_depreciation_time=True
                ):
                    if py in tech_sum_disc:
                        tech_sum_disc[py] += discount_factor[y]
            # capex parameter selection by tech type. The actual DataArray
            # dim names differ across tech types (e.g. conversion uses
            # 'level_0' for its tech dim, 'year' for years; storage/transport
            # use the canonical set names). For PWA conversion techs,
            # capex_specific_conversion is all-NaN/zero and the
            # finite-positive mask below leaves tilde_c_i at 0 (valid
            # under-approximation, just looser).
            if tech in set_conv:
                cs_full = params.capex_specific_conversion
                tech_dim = "level_0" if "level_0" in cs_full.dims else "set_technologies"
                cs = cs_full.sel({tech_dim: tech})
                year_dim_name = "year" if "year" in cs.dims else "set_time_steps_yearly"
            elif tech in set_stor:
                cs = params.capex_specific_storage.sel(set_storage_technologies=tech)
                year_dim_name = "set_time_steps_yearly"
            elif tech in set_tran:
                cs = params.capex_specific_transport.sel(set_transport_technologies=tech)
                year_dim_name = "set_time_steps_yearly"
            else:
                n_no_capex += 1
                continue
            # cs is a DataArray on remaining dims (cap_type/segment, loc/edge, year)
            cs_vals = cs.values
            if year_dim_name not in cs.dims:
                # parameter doesn't have a yearly axis — unexpected; skip Tier 2 for this tech
                continue
            year_axis = cs.dims.index(year_dim_name)
            year_coords = list(cs.coords[year_dim_name].values)
            disc_vec = np.array(
                [tech_sum_disc.get(int(yy), 0.0) for yy in year_coords]
            )
            # broadcast disc_vec to cs shape on the year axis
            disc_shape = [1] * cs_vals.ndim
            disc_shape[year_axis] = len(year_coords)
            disc_bc = disc_vec.reshape(disc_shape)
            coeff = cs_vals * a_h * disc_bc
            # min over finite, positive entries (NaN/0 mean parameter not set
            # for that tuple — for under-approximation we want a positive lower
            # bound on unit cost, taken over tuples where the capex actually
            # applies).
            finite_pos = np.isfinite(coeff) & (coeff > 0)
            if not finite_pos.any():
                if tech in set_conv:
                    # likely PWA conversion tech
                    n_pwa += 1
                tilde_c[idx] = 0.0
            else:
                tilde_c[idx] = float(coeff[finite_pos].min())

        # --- Assemble (A0, b0) ---
        neg_id = -np.eye(n_z)
        neg_id_b = np.zeros(n_z)
        rows = [neg_id]
        b_parts = [neg_id_b]
        if tier1_rows:
            rows.append(np.vstack(tier1_rows))
            b_parts.append(np.array(tier1_b))
        # Always emit the Tier 2 row even if tilde_c is all zero — the
        # constraint is vacuous (0 <= (1+eps) C*) but keeps row indexing
        # consistent and lets the user see in the diagnostic that the
        # cost-cap row was inactive.
        rows.append(tilde_c.reshape(1, -1))
        b_parts.append(np.array([(1.0 + self.epsilon) * self.c_star]))

        A0 = np.vstack(rows)
        b0 = np.concatenate(b_parts)

        # --- Sanity assert: A0 @ z* <= b0 (with tolerance) ---
        z_star = self._extract_z()
        lhs = A0 @ z_star
        abs_tol = 1e-6 * (np.abs(b0) + 1.0)
        violations = lhs > b0 + abs_tol
        if violations.any():
            i = int(np.argmax(lhs - b0))
            raise RuntimeError(
                f"MGA initial outer approximation violates containment of z*: "
                f"row {i}: lhs={lhs[i]:.6g} > rhs={b0[i]:.6g} "
                f"(slack {lhs[i] - b0[i]:.3g}). Tier 1 row count={len(tier1_rows)}; "
                f"if i < n_z this is a non-negativity row (z*_i < 0 — should be impossible); "
                f"if n_z <= i < n_z + tier1_count this is a Tier 1 row; "
                f"otherwise it is the Tier 2 cost-cap row."
            )

        # --- Diagnostic logging ---
        ratio = float(tilde_c @ z_star / self.c_star) if self.c_star else 0.0
        logging.info(
            f"MGA outer approximation: n_z={n_z}, Tier 1 rows={len(tier1_rows)} "
            f"(capacity_addition_max: {n_tier1_cam}, capacity_limit: {n_tier1_clim}), "
            f"Tier 2 rows=1. tilde_c @ z* / C* = {ratio:.4g} "
            f"(should be in (0, 1] for a well-formed under-approximation; "
            f"techs with non-zero tilde_c: {int((tilde_c > 0).sum())}/{n_z}; "
            f"PWA-conversion techs with tilde_c=0: {n_pwa}; "
            f"techs not classifiable as conv/stor/tran: {n_no_capex})."
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
        # on z_techs, where it carries -direction (sign flip because we
        # maximise direction^T z over Z_eps via min (-direction)^T z).
        tech_coord = self.cap_add.coords["set_technologies"]
        w = xr.DataArray(
            np.zeros(tech_coord.size),
            dims=("set_technologies",),
            coords={"set_technologies": tech_coord},
        )
        w.loc[{"set_technologies": list(self.z_techs)}] = -direction

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

        New variables (both on dim "set_technologies" with the FILTERED
        z_techs coord; this re-uses the dim name so that linopy aligns
        Sx and delta naturally via xarray):
            mga_oracle_delta: vector, free.
            mga_oracle_t:     scalar, lower=0.

        New constraints (paper's z_f variable is eliminated by substitution:
        Sx = z_f and z_O - z_f = delta combine into Sx - delta == z_O):
            mga_oracle_proj_eq: Sx - delta == trial    (RHS updated per iter)
            mga_oracle_t_pos:   delta - t*1 <= 0
            mga_oracle_t_neg: -delta - t*1 <= 0

        Call exactly once before the ORACLE loop.
        """
        z_coord = xr.DataArray(
            np.array(self.z_techs),
            dims="set_technologies",
            coords={"set_technologies": self.z_techs},
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

        # S*x as a LinearExpression on filtered set_technologies.
        Sx = self.cap_add.sel(set_technologies=self.z_techs).sum(
            ["set_capacity_types", "set_location", "set_time_steps_yearly"]
        )

        # Initial trial RHS: zero vector. Will be overwritten in find_nearest_point.
        zero_trial = xr.DataArray(
            np.zeros(self.n_z),
            dims="set_technologies",
            coords={"set_technologies": self.z_techs},
        )

        self.model.add_constraints(Sx - self.delta == zero_trial, name="mga_oracle_proj_eq")
        self.model.add_constraints(self.delta - self.t_var <= 0, name="mga_oracle_t_pos")
        self.model.add_constraints(-self.delta - self.t_var <= 0, name="mga_oracle_t_neg")
        logging.info(f"MGA oracle: projection model added (n_z = {self.n_z})")

    def find_nearest_point(self, trial_point: np.ndarray):
        """Callback for pyoNearOpt ORACLE.

        Returns (z_feas, dist, mu_cut, b_cut, flag) where mu_cut is L2-normalised
        and b_cut = mu_cut @ z_feas, matching the decagon reference convention.
        """
        if trial_point.shape != (self.n_z,):
            raise ValueError(
                f"find_nearest_point: expected shape ({self.n_z},), got {trial_point.shape}"
            )

        # Update only the RHS of the projection equality (in-place; cheap).
        trial_da = xr.DataArray(
            trial_point,
            dims="set_technologies",
            coords={"set_technologies": self.z_techs},
        )
        self.model.constraints["mga_oracle_proj_eq"].rhs = trial_da

        # Objective: min t. The 1 * t_var coerces the scalar Variable to a
        # LinearExpression, which is what add_objective expects.
        # Sum collapses the trivial mga_oracle_scalar_dim back to a scalar
        # expression for the objective.
        self.model.add_objective(self.t_var.sum(), sense="min", overwrite=True)

        label = f"oracle_iter_{self._iter_count}"
        logging.info(
            f"MGA oracle: starting iteration {self._iter_count}, "
            f"||trial_point||_2 = {np.linalg.norm(trial_point):.4g}"
        )
        self._solve_and_postprocess(label)

        z_feas = self._extract_z()

        # solution has shape (1,) on mga_oracle_scalar_dim; extract the scalar.
        dist = float(self.t_var.solution.values[0])

        # Dual on the projection equality. Same sign convention as scipy
        # linprog's eqlin.marginals (confirmed via notes/dual_sign_toy.py).
        mu_da = (
            self.model.constraints["mga_oracle_proj_eq"].dual
            .sel(set_technologies=self.z_techs)
            .values
        )
        # L2-normalise mu_cut (mirrors the decagon reference).
        scale = np.linalg.norm(mu_da, ord=2)
        if scale > 1e-4:
            mu_da = mu_da / scale
        b_cut = float(mu_da @ z_feas)

        logging.info(
            f"MGA oracle iter {self._iter_count}: dist = {dist:.4g}, "
            f"|mu|_max = {np.max(np.abs(mu_da)):.4g}, b_cut = {b_cut:.4g}"
        )
        self._iter_count += 1
        return z_feas, dist, mu_da, b_cut, 0


# ----------------------------------------------------------------------
# Event handler (mode dispatcher)
# ----------------------------------------------------------------------

@EventPublisher.register(Event.after_solve)
def run_mga(*args, **kwargs):
    """Entry point invoked by EventPublisher after the baseline solve."""
    mode = config.get("mode", "weights")
    epsilon = config["epsilon"]
    exclude_techs = config.get("exclude_techs", [])

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
        f"n_exclude = {len(exclude_techs)}"
    )

    mga = MGA(optimization_setup, epsilon, postprocess_ctx, exclude_techs=exclude_techs)
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
        from pyoNearOpt.exploration_methods.random import random_directions
        from pyoNearOpt.polytope_approximation.approximation_class import approximation

        rd_cfg = config.get("random_directions", {})
        n_iter = rd_cfg.get("n_iterations", 50)
        seed = rd_cfg.get("seed", None)
        if seed is not None:
            np.random.seed(seed)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Coordinates across variables not equal.*",
                category=UserWarning,
            )
            z_star = mga._extract_z()
            A0, b0 = mga.build_initial_outer_approximation()

            poly = approximation(
                A=A0, X=z_star.reshape(1, -1), b=b0,
                name_list=list(mga.z_techs),
                use_bigM=False,
            )

            rd = random_directions(support_function=mga.support_function, poly_approx=poly)
            rd.explore(n_iter=n_iter)

        # Summary folder lands inside folder_output, as a sibling of the
        # per-iteration Postprocess folders. postprocess_ctx["subfolder"] is
        # only a relative sub-path (empty for non-scenario runs), so using it
        # alone would resolve against the process cwd instead.
        out = (
            Path(optimization_setup.analysis.folder_output)
            / f"{postprocess_ctx['model_name']}_random_dir_summary"
        )
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "polytope.npz",
            A=poly.A, b=poly.b, X=poly.X,
            name_list=np.array(poly.name_list),
        )
        logging.info(f"MGA random_directions: complete. Polytope saved to {out}")

    elif mode == "oracle":
        from pyoNearOpt.exploration_methods.ORACLE import oracle as ORACLEAlgorithm
        from pyoNearOpt.polytope_approximation.approximation_class import approximation
        import pyomo.environ as pyo

        ora_cfg = config.get("oracle", {})
        max_iter = ora_cfg.get("max_iterations", 200)
        if "tolerance" not in ora_cfg:
            raise ValueError("MGA oracle: 'tolerance' is required.")
        tol = float(ora_cfg["tolerance"])
        Md_override = ora_cfg.get("Md_override", 1e8)
        t_max_override = ora_cfg.get("t_max_override", None)

        # Add projection model to the linopy problem (ONCE).
        mga.setup_projection_model()

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Coordinates across variables not equal.*",
                category=UserWarning,
            )
            z_star = mga._extract_z()
            A0, b0 = mga.build_initial_outer_approximation()
            logging.info(f"MGA oracle: tol = {tol:.3g}")

            poly = approximation(
                A=A0, X=z_star.reshape(1, -1), b=b0,
                name_list=list(mga.z_techs),
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
        pyomo_solver.set_options("OutputFlag=0")

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
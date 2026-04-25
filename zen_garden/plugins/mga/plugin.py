"""
MGA (Modeling to Generate Alternatives) plugin for ZEN-garden.

Registers an `after_solve` event handler that, after the baseline optimization
has been solved and written to disk, adds a near-optimality cost constraint
and re-solves the problem one or more times with alternative objectives of the
form g = sum_i w_i * x_i, where x_i are capacity_addition variables.

Each MGA iteration is written to disk as a sibling sub-solution next to the
baseline, using Postprocess with a modified model_name.

Configuration (via the "plugins.mga" block in config.json):
    epsilon (float): near-optimality slack; MGA solutions must satisfy
        cost <= (1 + epsilon) * C*, where C* is the baseline optimum.
    iterations (list[dict]): one entry per MGA iteration. Each entry has
        a "weights" key with a {technology_name: float} dict. Positive
        weights penalize a technology, negative weights promote it.
        Technologies not listed default to weight 0.
"""

import logging

import numpy as np
import xarray as xr

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.postprocess.postprocess import Postprocess


# Module-level config dict. The plugin loader merges user values from
# config.json's "plugins.mga" block into this dict via in-place update,
# so defaults here are preserved when the user omits a key.
config = {
    "epsilon": 0.1,
    "iterations": [],
}


class MGA:
    """MGA core logic.

    The split between `setup()` and `run_iteration()` exists because
    `linopy.Model.add_constraints` is not idempotent: calling it a second
    time with the same name raises ValueError. The near-optimality constraint
    is therefore added exactly once in `setup()`, while the objective is
    swapped per iteration in `run_iteration()` using `overwrite=True`.
    """

    def __init__(self, optimization_setup, epsilon, postprocess_ctx):
        """
        Args:
            optimization_setup: The OptimizationSetup instance, expected to
                already hold the solved baseline (so model.objective.value
                equals C*).
            epsilon: Near-optimality slack, e.g. 0.1 for a 10% cost budget.
            postprocess_ctx: Dict with keys "scenarios", "subfolder",
                "model_name", "scenario_name", "param_map". Forwarded to
                Postprocess for each iteration's output.
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

    def setup(self):
        """Add the near-optimality cost constraint. Call exactly once per run."""
        orig_cost_expr = self.optimization_setup.energy_system.rules.objective_total_cost(self.model)
        self.model.add_constraints(
            orig_cost_expr <= (1 + self.epsilon) * self.c_star,
            name="mga_near_optimality", # maybe problematic for multiple scenario runs?
        )
        logging.info(
            f"MGA: added near-optimality constraint cost <= "
            f"{(1 + self.epsilon) * self.c_star} "
            f"(C* = {self.c_star}, epsilon = {self.epsilon})"
        )

    def run_iteration(self, weights: dict, iter_id: int):
        """Run one MGA iteration.

        Replaces the model's objective with g = sum_i w_i * x_i, re-solves,
        and persists the result as a sibling sub-solution.

        Args:
            weights: {technology_name: float}. Missing technologies default
                to weight 0.
            iter_id: Integer used to form the output folder name.
        """
        w = self._build_w(weights)
        mga_obj = (w * self.cap_add).sum()
        self.model.add_objective(mga_obj, sense="min", overwrite=True)
        logging.info(f"MGA iter {iter_id}: objective replaced with weights = {weights}")

        self.optimization_setup.solve()
        logging.info(f"MGA iter {iter_id}: termination = {self.model.termination_condition}")

        base_name = self.postprocess_ctx["model_name"]
        Postprocess(
            self.optimization_setup,
            scenarios=self.postprocess_ctx["scenarios"],
            subfolder=self.postprocess_ctx["subfolder"],
            model_name=f"{base_name}_mga_iter_{iter_id}",
            scenario_name=self.postprocess_ctx["scenario_name"],
            param_map=self.postprocess_ctx["param_map"],
        )
        logging.info(f"MGA iter {iter_id}: written to sub-solution '{base_name}_mga_iter_{iter_id}'")

    def _build_w(self, weights: dict) -> xr.DataArray:
        """Build a 1D xarray DataArray over set_technologies from a weights dict.

        Raises KeyError for unknown technology names to catch typos early.
        xarray broadcasts the resulting 1D array across (cap_type, location,
        year) automatically when multiplied with capacity_addition.
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


@EventPublisher.register(Event.after_solve)
def run_mga(*args, **kwargs):
    """Entry point invoked by EventPublisher after the baseline solve.

    Reads `epsilon` and `iterations` from the module-level `config` dict
    (populated by the plugin loader from config.json). Does nothing if no
    iterations are configured — this allows the plugin to be installed
    and loaded without forcing every run to execute MGA.
    """
    iterations = config["iterations"]
    if not iterations:
        logging.warning("MGA plugin active but no iterations configured; skipping.")
        return

    epsilon = config["epsilon"]
    optimization_setup = kwargs["optimization_setup"]
    postprocess_ctx = {
        "scenarios": kwargs["scenarios"],
        "subfolder": kwargs["subfolder"],
        "model_name": kwargs["model_name"],
        "scenario_name": kwargs["scenario_name"],
        "param_map": kwargs["param_map"],
    }
    logging.info(f"MGA plugin: epsilon = {epsilon}, {len(iterations)} iteration(s)")

    mga = MGA(optimization_setup, epsilon, postprocess_ctx)
    mga.setup()
    for i, iteration in enumerate(iterations):
        if "weights" not in iteration:
            raise KeyError(f"MGA iteration {i} missing 'weights' key")
        weights = iteration["weights"]
        if not isinstance(weights, dict):
            raise TypeError(
                f"MGA iteration {i}: 'weights' must be a dict, got {type(weights).__name__}"
            )
        mga.run_iteration(weights, i)

    logging.info("MGA plugin: all iterations complete.")

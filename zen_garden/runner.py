"""This function runs ZEN garden,it is executed in the __main__.py script.
Compilation  of the optimization problem.
"""

import importlib
import importlib.util
import json
import logging
import os
import warnings
from pathlib import Path

import zen_garden.default_config as default_config

from .optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess
from .utils import InputDataChecks, ScenarioUtils, StringUtils, setup_logger

import numpy as np
import xarray as xr

# we setup the logger here
setup_logger()


def run(config="./config.json", dataset=None, job_index=None, folder_output=None):
    """Run ZEN-garden.

    This function is the primary programmatic entry point for running
    ZEN-garden. When called, it reads the configuration, loads the model
    input data, constructs and solves the optimization problem, and saves
    the results.

    Args:
        config (str): Path to the configuration file (e.g. ``config.json``).
            If the file is located in the current working directory, the
            filename alone may be specified. Defaults to ``"./config.json"``.
        dataset (str): Path to the folder containing the input dataset
            (e.g. ``"./1_base_case"``). If located in the current working
            directory, the folder name alone may be used. Defaults to the
            ``dataset`` value specified in the configuration file.
        folder_output (str): Path to the folder where outputs will be saved.
            Defaults to ``"./outputs"``.
        job_index (list[int] | None): Indices of jobs (scenarios) to run.
            For example, ``job_index=[1]`` runs only the first scenario.
            Defaults to ``None`` (run all jobs).

    Returns:
        OptimizationSetup: The fully set up and solved optimization problem.

    Examples:
        >>> from zen_garden import run, download_example_dataset
        >>> download_example_dataset("1_base_case")
        >>> run("1_base_case")
    """
    # print the version
    version = importlib.metadata.version("zen-garden")
    logging.info(f"Running ZEN-garden version: {version}")

    # prevent double printing
    logging.propagate = False

    ### import the config
    if not os.path.exists(config):
        config = config.replace(".py", ".json")
    config_path, config_file = os.path.split(os.path.abspath(config))
    if config_file.endswith(".py"):
        spec = importlib.util.spec_from_file_location(
            "module", Path(config_path) / config_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
        warnings.warn(
            "Use of the `config.py` file is deprecated and will be removed "
            "in ZEN-garden v3.0.0. Please switch to using a `config.json` "
            "file instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    else:
        with open(Path(config_path) / config_file, "r") as f:
            config = default_config.Config(**json.load(f))

    # overwrite the path if necessary
    if dataset is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis.dataset = dataset
    if folder_output is not None:
        if not Path(folder_output).is_absolute():
            folder_output = os.path.abspath(Path(config_path) / folder_output)
        config.analysis.folder_output = folder_output
        config.solver.solver_dir = folder_output
    logging.info(f"Optimizing for dataset {config.analysis.dataset}")
    # make all paths absolute to the config file path
    if not Path(config.analysis.dataset).is_absolute():
        config.analysis.dataset = os.path.abspath(
            Path(config_path) / config.analysis.dataset
        )
    if not Path(config.analysis.folder_output).is_absolute():
        config.analysis.folder_output = os.path.abspath(
            Path(config_path) / config.analysis.folder_output
        )
    if not Path(config.solver.solver_dir).is_absolute():
        config.solver.solver_dir = os.path.abspath(
            Path(config_path) / config.solver.solver_dir
        )
    config.analysis.zen_garden_version = version
    ### SYSTEM CONFIGURATION
    input_data_checks = InputDataChecks(config=config, optimization_setup=None)
    input_data_checks.check_dataset()
    input_data_checks.read_system_file(config)
    input_data_checks.check_technology_selections()
    input_data_checks.check_year_definitions()
    # overwrite default system and scenario dictionaries
    scenarios, elements = ScenarioUtils.get_scenarios(config, job_index)
    # get the name of the dataset
    model_name, out_folder = StringUtils.setup_model_folder(
        config.analysis, config.system
    )
    # clean sub-scenarios if necessary
    ScenarioUtils.clean_scenario_folder(config, out_folder)
    ### ITERATE THROUGH SCENARIOS
    for scenario, scenario_dict in zip(scenarios, elements, strict=False):
        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the scenario_dict and read input data
        optimization_setup = OptimizationSetup(
            config, scenario_dict=scenario_dict, input_data_checks=input_data_checks
        )
        # get rolling horizon years
        steps_horizon = optimization_setup.get_optimization_horizon()
        # iterate through horizon steps
        for step in steps_horizon:
            StringUtils.print_optimization_progress(
                scenario, steps_horizon, step, system=config.system
            )
            # overwrite time indices
            optimization_setup.overwrite_time_indices(step)
            # create optimization problem
            optimization_setup.construct_optimization_problem() #in energy_system.py line 524 (also creates objective function)
            if optimization_setup.solver.use_scaling:
                optimization_setup.scaling.run_scaling()
            elif (
                optimization_setup.solver.analyze_numerics
                or optimization_setup.solver.run_diagnostics
            ):
                optimization_setup.scaling.analyze_numerics()
            # SOLVE THE OPTIMIZATION PROBLEM
            optimization_setup.solve() # in optimization_setup.py line 676 (minimizes cost function)
            # break if infeasible
            if not optimization_setup.optimality:
                # write IIS
                optimization_setup.write_IIS(scenario)
                logging.warning(
                    f"Optimization: {optimization_setup.model.termination_condition}"
                )
                break
            if optimization_setup.solver.use_scaling:
                optimization_setup.scaling.re_scale()
            # save new capacity additions and cumulative carbon emissions
            # for next time step
            if optimization_setup.system.use_rolling_horizon:
                optimization_setup.add_results_of_optimization_step(step)
            # EVALUATE RESULTS
            # create scenario name, subfolder and param_map for postprocessing
            scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                config=config,
                scenario=scenario,
                scenario_dict=scenario_dict,
                steps_horizon=steps_horizon,
                step=step,
            )
            # write results
            Postprocess(
                optimization_setup,
                scenarios=config.scenarios,
                subfolder=subfolder,
                model_name=model_name,
                scenario_name=scenario_name,
                param_map=param_map,
            )

    # === MGA FEASIBILITY TEST ===

    # 1 Variable handle (linopy symbolic variable, not solution values)
    model = optimization_setup.model
    cap_add = model.variables["capacity_addition"]

    # 2 Capture baseline: optimal cost C* and the full baseline solution.
    #    The solution snapshot is needed BEFORE the second solve overwrites it.
    c_star = model.objective.value
    baseline_sol = model.solution["capacity_addition"]
    baseline_nuclear = baseline_sol.sel(set_technologies="nuclear").sum().item()
    baseline_pv      = baseline_sol.sel(set_technologies="photovoltaics").sum().item()
    logging.info(f"C* = {c_star}")
    logging.info(f"Baseline nuclear total: {baseline_nuclear}")
    logging.info(f"Baseline PV total:      {baseline_pv}")

    # 3 Near-optimality constraint: f(x) <= (1+eps) * C*.
    #    objective_total_cost(model) returns the original cost LinearExpression
    #    without setting it as the model's objective, so we can reuse it as a
    #    constraint while replacing the objective with the MGA one.
    epsilon = 0.1
    orig_cost_expr = optimization_setup.energy_system.rules.objective_total_cost(model)
    model.add_constraints(
        orig_cost_expr <= (1 + epsilon) * c_star,
        name="mga_near_optimality",
    )
    logging.info(f"Added constraint: cost <= {(1 + epsilon) * c_star}")

    # 4 Generic w-builder: takes a {tech_name: weight} dict and returns an
    #    xarray.DataArray indexed only by set_technologies. xarray broadcasts
    #    this across (cap_type, location, year) automatically when multiplied
    #    with capacity_addition. No hardcoded technology names inside.
    def build_w_from_dict(model, weights: dict) -> xr.DataArray:
        tech_coord = model.variables["capacity_addition"].coords["set_technologies"]
        w = xr.DataArray(
            np.zeros(tech_coord.size),
            dims=("set_technologies",),
            coords={"set_technologies": tech_coord},
        )
        known = set(tech_coord.values)
        for tech, val in weights.items():
            if tech not in known:
                raise KeyError(f"Unknown technology in weights: {tech!r}")
            w.loc[tech] = float(val)
        return w

    # 5 MGA objective: g = sum_i w_i * x_i, minimized.
    #    Convention: w_i > 0 penalizes tech i, w_i < 0 promotes it.
    #    overwrite=True is required to replace the original cost objective.
    weights = {"nuclear": -1.0, "photovoltaics": +1.0}
    w = build_w_from_dict(model, weights)
    mga_obj = (w * cap_add).sum()
    model.add_objective(mga_obj, sense="min", overwrite=True)
    logging.info(f"MGA weights: {weights}")
    logging.info("Objective replaced: minimize sum_i w_i * x_i")

    # 6 Re-solve with modified problem (constraint + new objective).
    optimization_setup.solve()
    logging.info(f"MGA termination: {model.termination_condition}")

    # 7 Validation (smoke-test specific; not part of the MGA feature itself):
    #    check that promoted/penalized techs moved as expected and that the
    #    cost stayed within the near-optimality bound.
    sol = model.solution["capacity_addition"]
    mga_nuclear = sol.sel(set_technologies="nuclear").sum().item()
    mga_pv      = sol.sel(set_technologies="photovoltaics").sum().item()
    mga_cost    = orig_cost_expr.solution.sum().item()

    logging.info("\n=== RESULTS ===")
    logging.info(f"Baseline nuclear: {baseline_nuclear}")
    logging.info(f"MGA nuclear (w=-1, should INCREASE): {mga_nuclear}")
    logging.info(f"Baseline PV: {baseline_pv}")
    logging.info(f"MGA PV (w=+1, should DECREASE): {mga_pv}")
    logging.info(f"MGA cost: {mga_cost}")
    logging.info(f"Cost bound (1+eps)*C*: {(1 + epsilon) * c_star}")
    logging.info(f"Cost feasible: {mga_cost <= (1 + epsilon) * c_star}")
    logging.info("=== MGA FEASIBILITY TEST COMPLETE ===")

    # 8 Persist MGA solution as a sibling subsolution.
    #    Using a modified model_name puts the MGA output in a parallel folder
    #    next to the baseline, which zen-visualization surfaces as a separate
    #    selectable subsolution.
    Postprocess(
        optimization_setup,
        scenarios=config.scenarios,
        model_name=model_name + "_mga_iter_0",
        subfolder=subfolder,
        scenario_name=scenario_name,
        param_map=param_map,
    )
    logging.info(f"MGA solution written to subsolution '{model_name}_mga_iter_0'")


    logging.info("--- Optimization finished ---")
    return optimization_setup

import pandas as pd
import numpy as np
from pathlib import Path
import importlib
import importlib.util
import json
import logging
import os
import warnings
from pathlib import Path

import zen_garden.default_config as default_config
from zen_garden.plugin_system.loader import register_plugins
from zen_garden.plugin_system.events import EventPublisher, Event

from zen_garden.optimization_setup import OptimizationSetup
from zen_garden.postprocess.postprocess import Postprocess
from zen_garden.utils import InputDataChecks, ScenarioUtils, StringUtils, setup_logger
from zen_garden.wrapper.utils import load_results


def check_for_semidefinite(correlation_matrix):

    # Check positive semidefiniteness of the correlation matrix
    corr_mat = correlation_matrix.values.astype(float)
    eigenvalues = np.linalg.eigvalsh(corr_mat)
    min_eigenvalue = eigenvalues.min()
    if min_eigenvalue < -1e-8:
        print(
            f"Correlation matrix is NOT positive semidefinite. "
            f"Minimum eigenvalue: {min_eigenvalue:.6g}. "
            "The variance term may be non-convex."
        )
    elif min_eigenvalue < 0:
        print(
            f"Correlation matrix is marginally non-PSD (min eigenvalue = {min_eigenvalue:.2e}), "
            "likely due to numerical noise — treating as PSD."
        )
    else:
        print(
            f"Correlation matrix is positive semidefinite (min eigenvalue = {min_eigenvalue:.6g})."
        )

def get_all_runs(run):
    result_path_linear = run / "linear"
    result_path_quadratic = run / "capex"


    output_dirs = {
        p.name: p
        for p in result_path_quadratic.iterdir()
        if p.is_dir() and not p.name.startswith("linear")
    }

    output_dirs["linear"] = result_path_linear

    return output_dirs

def create_cost_sample():

    # Absolute SD per technology
    absolute_sd_per_tech = calculate_absolute_sd(optimization_setup)

    # Correlation per technology pair
    corr_df = calculate_correlation_matrix(optimization_setup)

    # Build (tech_i, cap_i) × (tech_j, cap_j) pairs with correlation
    pairs = generate_covariance_pairs(absolute_sd_per_tech, corr_df)

class ModelApi:

    def __init__(self, config="./config.json", dataset=None, job_index=None, folder_output=None):

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

        register_plugins(config.plugins)

        # overwrite the path if necessary
        if dataset is not None:
            # logging.info(f"Overwriting dataset to: {dataset_path}")
            config.analysis.dataset = dataset

        self.dataset = dataset
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

        EventPublisher.trigger(Event.on_preprocessing, config)

        ### SYSTEM CONFIGURATION
        self.input_data_checks = InputDataChecks(config=config, optimization_setup=None)
        self.input_data_checks.check_dataset()
        self.input_data_checks.read_system_file(config)
        self.input_data_checks.check_technology_selections()
        self.input_data_checks.check_year_definitions()
        # overwrite default system and scenario dictionaries
        self.scenarios, self.elements = ScenarioUtils.get_scenarios(config, job_index)
        # get the name of the dataset
        self.model_name, out_folder = StringUtils.setup_model_folder(
            config.analysis, config.system
        )
        # clean sub-scenarios if necessary
        ScenarioUtils.clean_scenario_folder(config, out_folder)

        self.config = config
        self.optimization_setup = None

        self.scenario = None
        self.scenario_dict = None
        self.steps_horizon = None
        self.step = None


    def build_model(self):
        ### ITERATE THROUGH SCENARIOS

        if len(self.scenarios) != 1:
            Exception("Multiple scenarios not allowed!")
        for scenario, scenario_dict in zip(self.scenarios, self.elements, strict=False):
            # FORMULATE THE OPTIMIZATION PROBLEM
            # add the scenario_dict and read input data
            optimization_setup = OptimizationSetup(
                self.config, scenario_dict=scenario_dict, input_data_checks=self.input_data_checks
            )
            # get rolling horizon years
            steps_horizon = optimization_setup.get_optimization_horizon()

            if len(steps_horizon) != 1:
                Exception("Rolling horizon not allowed!")
            # iterate through horizon steps
            for step in steps_horizon:
                StringUtils.print_optimization_progress(
                    scenario, steps_horizon, step, system=self.config.system
                )
                # overwrite time indices
                optimization_setup.overwrite_time_indices(step)
                # create optimization problem
                optimization_setup.construct_optimization_problem()
                EventPublisher.trigger(Event.after_model_construction, optimization_setup=optimization_setup)

                self.optimization_setup = optimization_setup

                self.scenario = scenario
                self.scenario_dict = scenario_dict
                self.steps_horizon = steps_horizon
                self.step = step

    def solve_model(self):
        optimization_setup = self.optimization_setup

        if optimization_setup.solver.use_scaling:
            optimization_setup.scaling.run_scaling()
        elif (
                optimization_setup.solver.analyze_numerics
                or optimization_setup.solver.run_diagnostics
        ):
            optimization_setup.scaling.analyze_numerics()
        # SOLVE THE OPTIMIZATION PROBLEM
        optimization_setup.solve()
        # break if infeasible
        if not optimization_setup.optimality:
            # write IIS
            optimization_setup.write_IIS(self.scenario)
            logging.warning(
                f"Optimization: {optimization_setup.model.termination_condition}"
            )
        if optimization_setup.solver.use_scaling:
            optimization_setup.scaling.re_scale()
        # save new capacity additions and cumulative carbon emissions
        # for next time step
        if optimization_setup.system.use_rolling_horizon:
            optimization_setup.add_results_of_optimization_step(self.step)
        # EVALUATE RESULTS
        # create scenario name, subfolder and param_map for postprocessing
        scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
            config=self.config,
            scenario=self.scenario,
            scenario_dict=self.scenario_dict,
            steps_horizon=self.steps_horizon,
            step=self.step,
        )
        # write results
        Postprocess(
            optimization_setup,
            scenarios=self.config.scenarios,
            subfolder=subfolder,
            model_name=self.model_name,
            scenario_name=scenario_name,
            param_map=param_map,
        )
        logging.info("--- Optimization finished ---")

    def fix_capacities(self, path_to_capacities_to_fix_to):

        results = load_results(path_to_capacities_to_fix_to / self.dataset, None)
        capacity_addition = self.optimization_setup.model.variables["capacity_addition"]

        for (tech, loc, year), row in results["capacity_addition"].iterrows():
            for cap_type in ["energy", "power"]:

                value = row[cap_type]

                if pd.notna(value):
                    sel = dict(
                        set_technologies=tech,
                        set_capacity_types=cap_type,
                        set_location=loc,
                        set_time_steps_yearly=0,  # adapt if needed
                    )

                    capacity_addition.lower.loc[sel] = value - (1+10e-1)
                    capacity_addition.upper.loc[sel] = value + (1+10e-1)

    def fix_all_variables(self):

        self.optimization_setup.model.variables.fix()





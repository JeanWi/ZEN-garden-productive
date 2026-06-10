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
import linopy as lp
import pandas as pd
import psutil
import xarray as xr
from linopy.expressions import LinearExpression

import zen_garden.default_config as default_config
from zen_garden.plugin_system.loader import register_plugins
from zen_garden.plugin_system.events import EventPublisher, Event

from zen_garden.optimization_setup import OptimizationSetup
from zen_garden.postprocess.postprocess import Postprocess
from zen_garden.utils import InputDataChecks, ScenarioUtils, StringUtils, setup_logger
from zen_garden.wrapper.utils import load_results
from zen_garden.model.element import Element
from zen_garden.model.component import IndexSet
from zen_garden.utils import linexpr_from_tuple_np


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


def build_covariance_matrix(
    pairs: pd.DataFrame,
    absolute_sd_per_tech: pd.Series,
) -> tuple[pd.DataFrame, list[tuple]]:
    """Build a full covariance matrix from pairs and per-technology standard deviations.

    Args:
        pairs: DataFrame with columns [tech_i, cap_i, tech_j, cap_j, correlation].
        absolute_sd_per_tech: Series indexed by (set_technologies, set_capacity_types).

    Returns:
        cov_df: Square covariance matrix as a DataFrame, index/columns are (tech, cap) tuples.
        keys: Ordered list of (tech, cap) tuples corresponding to rows/columns.
    """
    keys = list(absolute_sd_per_tech.index)
    n = len(keys)
    key_idx = {k: i for i, k in enumerate(keys)}

    cov = np.zeros((n, n))

    for _, row in pairs.iterrows():
        i = key_idx.get((row["tech_i"], row["cap_i"]))
        j = key_idx.get((row["tech_j"], row["cap_j"]))
        sigma_i = absolute_sd_per_tech[(row["tech_i"], row["cap_i"])]
        sigma_j = absolute_sd_per_tech[(row["tech_j"], row["cap_j"])]
        cov[i, j] = row["correlation"] * sigma_i * sigma_j
        cov[j, i] = cov[i, j]  # ensure symmetry

    # Fill diagonal explicitly (σ² for each asset)
    for i, k in enumerate(keys):
        cov[i, i] = absolute_sd_per_tech[k] ** 2

    cov_df = pd.DataFrame(cov, index=pd.MultiIndex.from_tuples(keys), columns=pd.MultiIndex.from_tuples(keys))
    return cov_df, keys


def generate_samples(
    pairs: pd.DataFrame,
    absolute_sd_per_tech: pd.Series,
    n_samples: int = 1000,
    mean: pd.Series | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample N realisations from the multivariate normal capex distribution.

    Args:
        pairs: DataFrame with columns [tech_i, cap_i, tech_j, cap_j, correlation].
        absolute_sd_per_tech: Series indexed by (set_technologies, set_capacity_types),
            representing absolute standard deviations (σ = relative_sd × capex_specific).
        n_samples: Number of samples to draw.
        mean: Optional mean vector as a Series with the same index as absolute_sd_per_tech.
            Defaults to zero (i.e. samples represent deviations from expected cost).
        seed: Optional random seed for reproducibility.

    Returns:
        DataFrame of shape (n_samples, n_assets) where columns are (tech, cap) tuples.
    """
    cov_df, keys = build_covariance_matrix(pairs, absolute_sd_per_tech)
    cov_np = cov_df.values

    if mean is None:
        mean_np = np.zeros(len(keys))
    else:
        mean_np = np.array([float(mean[k]) for k in keys])

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean_np, cov_np, size=n_samples)

    col_idx = pd.MultiIndex.from_tuples(keys, names=["set_technologies", "set_capacity_types"])
    return pd.DataFrame(samples, columns=col_idx)


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

        self.capex_specific_storage = None
        self.capex_specific_transport = None
        self.capex_specific_conversion = None


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
                self.capex_specific_storage = self.optimization_setup.parameters.capex_specific_storage.copy()
                self.capex_specific_transport = self.optimization_setup.parameters.capex_specific_transport.copy()
                self.capex_specific_conversion = self.optimization_setup.parameters.capex_specific_conversion.copy()

    def solve_model(self, skip_postprocess = False, skip_scaling = False):
        optimization_setup = self.optimization_setup

        if not skip_scaling:
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

        if not skip_postprocess:
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

    def _fix_variables(self, variable_names):
        for var in variable_names:
            variable = self.optimization_setup.model.variables[var]

            variable.lower = variable.solution
            variable.upper = variable.solution

    def fix_design_variables(self):
        fix_vars = [
            "capacity_addition",
        ]

        self._fix_variables(fix_vars)

    def fix_operational_variables(self):
        fix_vars = [
            "flow_import",
            "flow_export",
            "flow_conversion_input",
            "flow_conversion_output",
            "flow_storage_charge",
            "flow_storage_discharge",
            "flow_transport",
            "carbon_emissions_technology"
        ]

        self._fix_variables(fix_vars)


    def reconstruct_cost_constraints(self, sample, demand_shedding_allowed = False):

        self._reconstruct_storage_cost_constraints(sample)
        self._reconstruct_transport_cost_constraints(sample)
        self._reconstruct_conversion_cost_constraints(sample)
        self._reconstruct_demand_shedding_constraint(demand_shedding_allowed)

    def _align_and_mask(self, expr, mask):
        """Aligns and masks expr.

        :param expr: expression to align and mask
        :param mask: mask to apply
        """
        if isinstance(expr, xr.DataArray):
            aligner = expr
        elif isinstance(expr, lp.Variable):
            aligner = expr.lower
        else:
            aligner = expr.const
        mask = xr.align(mask, aligner, join="right")[0]
        expr = expr.where(mask)
        return expr

    def _reconstruct_storage_cost_constraints(self, sample):

        self.optimization_setup.model.remove_constraints("constraint_storage_technology_capex")
        index_values, index_names = Element.create_custom_set(
            [
                "set_storage_technologies",
                "set_capacity_types",
                "set_nodes",
                "set_time_steps_yearly",
            ],
            self.optimization_setup,
        )

        ### auxiliary calculations
        # get all the arrays and coords
        techs, capacity_types, nodes, times = IndexSet.tuple_to_arr(
            index_values, index_names, unique=True
        )
        coords = [
            self.optimization_setup.model.variables.coords["set_storage_technologies"],
            self.optimization_setup.model.variables.coords["set_capacity_types"],
            self.optimization_setup.model.variables.coords["set_nodes"],
            self.optimization_setup.model.variables.coords["set_time_steps_yearly"],
        ]

        capex_specific_storage_original = self.capex_specific_storage.copy()

        tech_dim = "set_storage_technologies"
        techs = capex_specific_storage_original.coords[tech_dim].values
        caps = capex_specific_storage_original.coords["set_capacity_types"].values

        # Build a 2D DataArray (tech × cap_type) from the sample, aligned to the xarray coords
        delta = xr.DataArray(
            [[float(sample.get((t, c), 0.0)) for c in caps] for t in techs],
            dims=[tech_dim, "set_capacity_types"],
            coords={tech_dim: techs, "set_capacity_types": caps},
        )

        capex_specific_storage = capex_specific_storage_original + delta.broadcast_like(capex_specific_storage_original)



        ### formulate constraint
        lhs = linexpr_from_tuple_np(
            [
                (
                    1.0,
                    self.optimization_setup.model.variables["cost_capex_overnight"].loc[
                        techs, capacity_types, nodes, times
                    ],
                ),
                (
                    -capex_specific_storage.loc[
                        techs, capacity_types, nodes, times
                    ],
                    self.optimization_setup.model.variables["capacity_addition"].loc[
                        techs, capacity_types, nodes, times
                    ],
                ),
            ],
            coords,
            self.optimization_setup.model,
        )
        rhs = 0

        self.optimization_setup.model.add_constraints(lhs, "==", rhs, name="constraint_storage_technology_capex")


    def _reconstruct_conversion_cost_constraints(self, sample):

        self.optimization_setup.model.remove_constraints("constraint_linear_capex")


        capex_specific_conversion_original = self.capex_specific_conversion.copy()
        capex_specific_conversion_original = capex_specific_conversion_original.rename(
            {
                old: new
                for old, new in zip(
                list(capex_specific_conversion_original.dims),
                [
                    "set_conversion_technologies",
                    "set_nodes",
                    "set_time_steps_yearly",
                ],
                strict=False,
            )
            }
        )


        tech_dim = "set_conversion_technologies"
        techs = capex_specific_conversion_original.coords[tech_dim].values

        delta = xr.DataArray(
            [float(sample.get((t, "energy"), 0.0)) for t in techs],
            dims=[tech_dim],
            coords={tech_dim: techs},
        )

        capex_specific_conversion = capex_specific_conversion_original + delta.broadcast_like(capex_specific_conversion_original)


        capex_specific_conversion = capex_specific_conversion.broadcast_like(
            self.optimization_setup.model.variables["capacity_approximation"].lower
        )
        mask = ~np.isnan(capex_specific_conversion)
        lhs = lp.merge(
            [
                1 * self.optimization_setup.model.variables["capex_approximation"],
                -capex_specific_conversion * self.optimization_setup.model.variables["capacity_approximation"],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        lhs = self._align_and_mask(lhs, mask)
        rhs = 0

        self.optimization_setup.model.add_constraints(lhs, "==", rhs, name="constraint_linear_capex")


    def _reconstruct_transport_cost_constraints(self, sample):

        self.optimization_setup.model.remove_constraints("constraint_transport_technology_capex")

        index_values, index_list = Element.create_custom_set(
            ["set_transport_technologies", "set_edges", "set_time_steps_yearly"],
            self.optimization_setup,
        )
        # check if we even need to continue
        if len(index_values) == 0:
            return []
        # get the coords
        coords = [
            self.optimization_setup.parameters.capex_per_distance_transport.coords[
                "set_transport_technologies"
            ],
            self.optimization_setup.parameters.capex_per_distance_transport.coords["set_edges"],
            self.optimization_setup.parameters.capex_per_distance_transport.coords[
                "set_time_steps_yearly"
            ],
        ]

        ### masks
        # This mask checks the distance between nodes for the condition
        mask = np.isinf(self.optimization_setup.parameters.distance).astype(float)

        # This mask ensure we only get constraints where we want them
        index_arrs = IndexSet.tuple_to_arr(index_values, index_list)
        global_mask = xr.DataArray(False, coords=coords)
        global_mask.loc[index_arrs] = True

        capex_specific_transport_original = self.capex_specific_transport.copy()
        tech_dim = "set_transport_technologies"
        techs = capex_specific_transport_original.coords[tech_dim].values

        delta = xr.DataArray(
            [float(sample.get((t, "power"), 0.0)) for t in techs],
            dims=[tech_dim],
            coords={tech_dim: techs},
        )

        capex_specific_transport = capex_specific_transport_original + delta.broadcast_like(capex_specific_transport_original)


        ### auxiliary calculations TODO improve
        term_distance_inf = (
                mask
                * self.optimization_setup.model.variables["capacity_addition"].loc[
                    coords[0], "power", coords[1], coords[2]
                ]
        )
        term_distance_not_inf = (1 - mask) * (
                self.optimization_setup.model.variables["cost_capex_overnight"].loc[
                    coords[0], "power", coords[1], coords[2]
                ]
                - self.optimization_setup.model.variables["capacity_addition"].loc[
                    coords[0], "power", coords[1], coords[2]
                ]
                * capex_specific_transport.loc[coords[0], coords[1]]
        )
        # Additional check to avoid binary variables when their coefficient is 0
        if np.any(
                self.optimization_setup.parameters.distance.loc[coords[0], coords[1]]
                * self.optimization_setup.parameters.capex_per_distance_transport.loc[coords[0], coords[1]]
                != 0
        ):
            term_distance_not_inf -= (
                    (1 - mask)
                    * self.optimization_setup.model.variables["technology_installation"].loc[
                        coords[0], "power", coords[1], coords[2]
                    ]
                    * (
                            self.optimization_setup.parameters.distance.loc[coords[0], coords[1]]
                            * self.optimization_setup.parameters.capex_per_distance_transport.loc[
                                coords[0], coords[1]
                            ]
                    )
            )

        # formulate constraint
        lhs = term_distance_inf + term_distance_not_inf
        lhs = lhs.where(global_mask)
        rhs = 0

        self.optimization_setup.model.add_constraints(lhs, "==", rhs, name="constraint_transport_technology_capex")

    def _reconstruct_demand_shedding_constraint(self, allow_demand_shedding):

        self.optimization_setup.model.remove_constraints("constraint_cost_shed_demand")
        self.optimization_setup.model.remove_constraints("constraint_limit_shed_demand")

        if allow_demand_shedding:
            # cost of shedding demand
            lhs_cost = (
                    self.optimization_setup.model.variables["cost_shed_demand"]
                    - 100 * self.optimization_setup.model.variables[
                        "shed_demand"]
            )
            rhs_cost = 0

            # limit of shedding demand:
            #   either the demand (price != inf) or zero (price == inf)
            lhs_shed_demand = self.optimization_setup.model.variables["shed_demand"]
            rhs_shed_demand = 1

            print("I am here")

        else:
            mask = self.optimization_setup.parameters.price_shed_demand != np.inf

            # cost of shedding demand
            lhs_cost = (
                self.optimization_setup.model.variables["cost_shed_demand"]
                - self.optimization_setup.parameters.price_shed_demand * self.optimization_setup.model.variables["shed_demand"]
            ).where(mask)
            rhs_cost = 0

            # limit of shedding demand:
            #   either the demand (price != inf) or zero (price == inf)
            lhs_shed_demand = self.optimization_setup.model.variables["shed_demand"]
            rhs_shed_demand = self.optimization_setup.parameters.demand.where(mask, 0.0)

        self.optimization_setup.model.add_constraints(lhs_shed_demand, "<=", rhs_shed_demand, name="constraint_limit_shed_demand")
        self.optimization_setup.model.add_constraints(lhs_cost, "==", rhs_cost, name="constraint_cost_shed_demand")

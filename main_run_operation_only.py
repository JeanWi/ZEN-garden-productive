import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from preprocessing.helpers import get_all_runs, ModelApi
from zen_garden import run
from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.postprocess.postprocess import Postprocess

# SETTINGS
test_dataset = True
on_server = False
design_run = "outputs_20260609-104915_good_run_1periods"
design_path = Path("C:/ZenGardenInput/ZEN-models")

# PATHS
if on_server:
    root_path = Path("/home/jwiegner/ZEN-models")
else:
    if test_dataset:
        root_path = Path(r"C:\ZenGardenInput\example_datasets")
        dataset = "8_yearly_variation"
        os.chdir(root_path)
        design_run = "outputs_20260605-105726_8_yearly_variation_single_period_weight_factor_moved"

    else:
        root_path = Path("C:/ZenGardenInput/ZEN-models")
        os.chdir(root_path / "data")
        dataset = "Crystal_Ball"


# Sample
# sample = create_cost_sample()

# Sample from costs -> We need to run operation only once as the operational costs don't change,
# but we need to recalculate the total costs for each sample
# Build model

all_runs = get_all_runs(design_path / "data" / design_run)
for key, path_to_design_run in all_runs.items():

    # set variables
    result_folder = f"./{design_run}_operation/{key}"
    print(result_folder)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    with open("./config.json") as f:
        config = json.load(f)
    config["solver"]["solver_options"]["LogFile"] = f"{result_folder}/solver_design.log"
    # config["solver"]["solver_options"]["FeasibilityTol"] = 1e-2

    with open("./config_operation.json", "w") as f:
        json.dump(config, f, indent=4)

    # Model API
    m_api = ModelApi(config="./config_operation.json", dataset=dataset, folder_output=result_folder)
    m_api.build_model()

    A_matrix = m_api.optimization_setup.model.constraints.to_matrix()
    x = m_api.optimization_setup.model.constraints.vars

    vlabels = m_api.optimization_setup.model.variables.carbon_emissions_annual.fix()


    # solve operation
    m_api.solve_model()

    # Fix all variables
    m_api.fix_all_variables()

    # Adapt costs (loop)
    m_api.solve_model()

    # Fix capacities
    # m_api.fix_capacities(path_to_design_run)


    # Fix all variables
    # m_api.fix_all_variables()

    # Adapt costs (loop)
    # m_api.solve_model()

    # Report total cost


# optimization_setup = run(
#     config="./config.json", dataset=dataset, folder_output=result_folder
# )

# fix capacity expansions


# adapt capex parameters
#
# # rerun
# result_folder = f"./outputs_{time_str}_operation"
#
# if not os.path.exists(result_folder):
#     os.makedirs(result_folder)
#
# optimization_setup.solver.solver_options["LogFile"] = (
#     f"{result_folder}/solver_operation.log"
# )
# optimization_setup.solver.solver_dir = result_folder
# optimization_setup.analysis.folder_output = result_folder
#
# EventPublisher.trigger(
#     Event.after_model_construction, optimization_setup=optimization_setup
# )
#
# if optimization_setup.solver.use_scaling:
#     optimization_setup.scaling.run_scaling()
# elif (
#     optimization_setup.solver.analyze_numerics
#     or optimization_setup.solver.run_diagnostics
# ):
#     optimization_setup.scaling.analyze_numerics()
# # SOLVE THE OPTIMIZATION PROBLEM
# optimization_setup.solve()
#
# if optimization_setup.solver.use_scaling:
#     optimization_setup.scaling.re_scale()
#
# # EVALUATE RESULTS
# scenarios = {"": {}}
# subfolder = Path(".")
# model_name = dataset
# scenario_name = None
# param_map = None
# # write results
# Postprocess(
#     optimization_setup,
#     scenarios=scenarios,
#     subfolder=subfolder,
#     model_name=model_name,
#     scenario_name=scenario_name,
#     param_map=param_map,
# )

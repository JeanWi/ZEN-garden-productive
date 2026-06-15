import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from preprocessing.helpers import get_all_runs, ModelApi, generate_samples
from zen_garden import run
from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.postprocess.postprocess import Postprocess
from zen_garden.plugins.mean_variance_optimization.helpers import *


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

result_folder = f"./{design_run}_operation/"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

with open("./config.json") as f:
    config = json.load(f)
config["solver"]["solver_options"]["LogFile"] = f"{result_folder}/solver_operation.log"

with open("./config_operation.json", "w") as f:
    json.dump(config, f, indent=4)

# Model API
m_api = ModelApi(config="./config_operation.json", dataset=dataset, folder_output=result_folder)
m_api.build_model()

# solve full model
m_api.solve_model()

m_api.fix_design_variables()
m_api.fix_operational_variables()
m_api.reconstruct_cost_constraints(row)
m_api.solve_model()


objective = {}

for key, path_to_design_run in all_runs.items():
    objective[key] = []
    # set variables

    # Fix variables
    m_api.fix_design_variables()
    m_api.fix_operational_variables()

    # Absolute SD per technology
    absolute_sd_per_tech = _calculate_absolute_sd(m_api.optimization_setup)

    # Correlation per technology pair
    corr_df = calculate_correlation_matrix(m_api.optimization_setup)

    # Sample
    pairs = generate_covariance_pairs(absolute_sd_per_tech, corr_df)
    sample = generate_samples(pairs, absolute_sd_per_tech, n_samples = 2)

    capex_specific = get_capex_specific(m_api.optimization_setup).to_series()

    for index, row in sample.iterrows():

        m_api.reconstruct_cost_constraints(row)

        m_api.solve_model()

        total_cost = m_api.optimization_setup.model.objective.value

        objective[key].append(total_cost)


pd.DataFrame(objective).to_csv(f"./{design_run}_operation/objective_samples.csv", index=False)

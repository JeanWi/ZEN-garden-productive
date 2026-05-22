import os
import json
from statistics import covariance

import pandas as pd
from pathlib import Path

from zen_garden import run
from datetime import datetime
from preprocessing.helpers import check_for_semidefinite

on_server = True
if on_server:
    root_path = Path("/home/jwiegner/ZEN-models")
else:    
    root_path = Path("C:/ZenGardenInput/ZEN-models")

# Check for positive semi-definite matrix in input
tech_capex_path = root_path / Path("data/Crystal_Ball/mean_variance/technology_capex")
correlation = pd.read_csv(tech_capex_path / "correlation.csv", index_col=0)
check_for_semidefinite(correlation)

example_dataset = False
# weights = [0.0001]
weights = [10e6, 10e3, 1, 10e-3, 10e-6]
# weights = [10e6]
if example_dataset:
    os.chdir(root_path / "data/example_datasets")
    dataset = "8_yearly_variation"
else:
    os.chdir(root_path / "data")
    dataset = "Crystal_Ball"

now = datetime.now()
time_str = now.strftime("%Y%m%d-%H%M%S")
result_folder = f"./outputs_{time_str}/linear"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

with open("./config.json") as f:
    config = json.load(f)
config["solver"]["solver_options"]["LogFile"] = f"{result_folder}/solver.log"

with open("./config.json", "w") as f:
    json.dump(config, f, indent=4)

run(config="./config.json", dataset=dataset, folder_output=result_folder)

include_var_for = {
    "capex": ["technology_capex"],
    # "capex_opex": ["capex", "opex"],
    # "capex_opex_import_export": ["capex", "opex", "import", "export"],
    # "all": ["technology_capex", "technology_opex", "import", "export", "demand_shedding"]
}


for variance_inclusion in include_var_for.keys():

    for weight in weights:

        result_folder = f"./outputs_{time_str}/{variance_inclusion}/lambda_{str(weight)}"

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        with open("./config.json") as f:
            config = json.load(f)

        config["plugins"]["mean_variance_optimization"] = {}
        config["plugins"]["mean_variance_optimization"]["weighting_factor"] = weight
        config["plugins"]["mean_variance_optimization"]["include_variances_for"] = include_var_for[variance_inclusion]
        config["solver"]["solver_options"]["LogFile"] =  f"{result_folder}/solver.log"

        with open("./config_quadratic.json", "w") as f:
            json.dump(config, f, indent=4)

        run(
            config="./config_quadratic.json",
            dataset=dataset,
            folder_output=result_folder,
        )

        # Check for positive semi-definite matrix after preprocessing
        processed_covar = Path("C:/ZenGardenInput/ZEN-models/data/quadratic_term_log.csv")
        covar_long = pd.read_csv(processed_covar)

        covariance_col = "scalar_coeff (wf*corr*sigma_i*sigma_j)"
        covariance_matrix = covar_long.pivot(index="C_i_var", columns="C_j_var", values=covariance_col)

        # Reindex so rows and columns share the same ordered labels (union of both)
        all_vars = sorted(set(covariance_matrix.index) | set(covariance_matrix.columns))
        covariance_matrix = covariance_matrix.reindex(index=all_vars, columns=all_vars).fillna(0.0)

        check_for_semidefinite(covariance_matrix)
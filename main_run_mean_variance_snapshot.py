from datetime import datetime
import json
from pathlib import Path
import os
import pandas as pd
from scipy import sparse
import argparse

from zen_garden.plugins.mean_variance_optimization.plugin import config, construct_mean_variance_objective
from zen_garden.wrapper.utils import modify_json
from preprocessing.helpers import ModelApi, construct_model, generate_samples

# SETTINGS
example_dataset = False

nr_timesteps = 1

def main(task_id: int,
            weight: float,
            nr_time_steps: int = 1,
            run_on: str = "local",
            with_diagonal_variance_only: bool = False,
            include_variance_for_technology_capex: bool = False,
            include_variance_for_imports: bool = False,
            method = "weighting_factor"
         ):
    include_variances_for = []
    result_string = ""
    if include_variance_for_imports:
        include_variances_for.append("imports")
        result_string = result_string + "_imports"
    if include_variance_for_technology_capex:
        include_variances_for.append("technology_capex")
        result_string = result_string + "_technologycapex"
    if with_diagonal_variance_only:
        result_string = result_string + "_nocorrelation"


    now = datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M%S")

    print(f"Running task_id={task_id}, weight={weight}")

    # PATHS
    # load_settings
    with open('run_settings.json') as json_file:
        settings = json.load(json_file)

    root_path = Path(settings[run_on]['root_path'])
    if example_dataset:
        root_path = Path(f"C:\ZenGardenInput\example_datasets")
        dataset = "8_yearly_variation"
        os.chdir(Path(f"C:\ZenGardenInput\example_datasets"))

        modify_json(
            root_path / dataset / "system.json",
            {
                "aggregated_time_steps_per_year": nr_time_steps,
                "reference_year": 2023,
                "optimized_years": 1,
                "interval_between_years": 1,
                "conduct_time_series_aggregation": True
            },
        )
    else:
        dataset = "Crystal_Ball"
        os.chdir(root_path)

        modify_json(
            root_path / dataset / "system.json",
            {
                "aggregated_time_steps_per_year": nr_time_steps,
                "reference_year": 2050,
                "optimized_years": 1,
                "interval_between_years": 1,
                "conduct_time_series_aggregation": True
            },
        )

    # generate result folder
    results_root = f"./outputs_{time_str}_snapshot_{method}_T{str(nr_time_steps)}{result_string}"
    if not os.path.exists(results_root):
        os.makedirs(results_root, exist_ok=True)

    sample = pd.read_pickle(root_path / f"sample_T{str(nr_timesteps)}.pkl")

    sample = sample[0:100]

    result_folder = f"{results_root}/lambda_{str(weight)}"
    config["method"] = method
    if with_diagonal_variance_only:
        config["include_correlation"] = False
    m_api = construct_model(weight, task_id, dataset, result_folder, include_variances_for)
    m_api.solve_model()
    m_api.reevaluate_objective(result_folder, sample, include_variances_for, parallelize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--nr_time_steps", type=int, default=1)
    parser.add_argument("--run_on", type=str, default="local", choices=["local", "epse_server", "euler"])
    args = parser.parse_args()

    task_id = args.task_id

    if example_dataset:
        run_array = pd.read_excel("Run_array_test_case.xlsx", index_col=0)
    else:
        run_array = pd.read_excel("Run_array.xlsx", index_col=0)
    this_run = run_array.loc[task_id]

    main(
        task_id=args.task_id,
        weight=this_run['weight'],
        nr_time_steps=args.nr_time_steps,
        run_on=args.run_on,
        with_diagonal_variance_only=this_run['with_diagonal_variance_only'],
        include_variance_for_technology_capex=this_run['include_variance_for_technology_capex'],
        include_variance_for_imports=this_run['include_variance_for_imports'],
        method=this_run['method'],
    )
